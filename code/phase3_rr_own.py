"""Phase 3 rhetorical roles, built on the four Phase 1 models.

Every sentence is read by a Phase 1 model (own byte-level BPE, own weights)
and mean-pooled into one vector. A BiLSTM reads the sentence vectors of a
whole judgment in order, and a linear-chain CRF picks one of the 13 roles for
each sentence. The score is macro-F1 over all 13 roles, so a role that never
gets predicted counts as zero.

Adaptation of the Phase 1 model (--method):
    frozen   Phase 1 weights fixed; sentence vectors cached once. Baseline.
    lorma    LoRMA+  h = (I + (alpha/r) B A) W x
    lora     LoRA    h = W x + (alpha/r) B A x
    rslora   LoRA with the rank-stabilised scale alpha/sqrt(r)
    ia3      IA3: learned vectors rescale keys, values, and the
             feed-forward activation; nothing else in the Phase 1 model moves
    distill  a student (--student_method, LoRA by default) learns from the
             gold roles and from the role scores of a trained teacher
             (--teacher), usually the efficient LoRMA run

    python phase3_rr_own.py --arch efficient --method lorma --out <folder>
    python phase3_rr_own.py --arch encdec --method distill --teacher <efficient lorma>/best.pt --out <folder>

The best epoch by held-out macro-F1 writes <out>/submission.zip, which holds
prediction.json: {document id: [one role per sentence]}.
"""

import argparse
import json
import math
import os
import random
import time
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from legal_bpe import LegalBPE
from phase1_models import build_model
from phase2_lner import LoRALinear, adapt_linears, enable_full_attention, hidden_states

LOCAL = r"C:\Users\Anshuman Raj\CS781"
DEFAULT_WEIGHTS = {
    "efficient": os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\model.pt"),
    "decoder": os.path.join(LOCAL, r"local_weights\cs781_phase1_decoder\model.pt"),
    "hybrid": os.path.join(LOCAL, r"local_weights\cs781_phase1_hybrid\model.pt"),
    "encdec": os.path.join(LOCAL, r"encdec_run2\cs781_phase1_encdec\model.pt"),
}
DEFAULT_BPE = os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\legal_bpe.json")
DEFAULT_DATA = os.path.join(LOCAL, "phase3data")
TRAIN_FILES = ("Train/CL_train.json", "Train/IT_train.json")
DEV_FILES = ("Train/CL_dev.json", "Train/IT_dev.json")
TEST_FILES = ("Test/CL_test.json", "Test/IT_test.json")
DEFAULT_LR = {"lorma": 2e-4, "lora": 2e-4, "rslora": 2e-4, "ia3": 3e-3, "frozen": 0.0}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_vocab(data_dir):
    path = os.path.join(data_dir, "Phase_3_RR", "Train", "label_vocab.json")
    with open(path, encoding="utf-8") as handle:
        mapping = json.load(handle)
    return [label for label, _index in sorted(mapping.items(), key=lambda item: item[1])]


def load_docs(data_dir, names):
    docs = []
    for name in names:
        path = os.path.join(data_dir, "Phase_3_RR", *name.split("/"))
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        domain = "CL" if "/CL_" in name else "IT"
        if isinstance(data, dict):
            for doc_id, sentences in data.items():
                docs.append({"id": doc_id, "domain": domain, "text": list(sentences), "labels": None})
        else:
            for item in data:
                docs.append(
                    {"id": str(item["id"]), "domain": domain, "text": list(item["text"]), "labels": list(item["labels"])}
                )
    return docs


def tokenize_docs(tokenizer, docs, max_tokens):
    for doc in docs:
        doc["ids"] = [tokenizer.encode(sentence)[:max_tokens] or [tokenizer.unk_token_id] for sentence in doc["text"]]


class IA3Linear(nn.Module):
    """A frozen Phase 1 matrix with one learned scaling vector, starting at 1.

    On keys and values the vector scales the output. On the feed-forward
    down projection it scales the input, which is the activation after the
    non-linearity. For a fused query-key-value matrix the query part is kept
    at 1 through the mask.
    """

    def __init__(self, base, size, on_input, keep=None):
        super().__init__()
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.on_input = on_input
        self.scale = nn.Parameter(torch.ones(size))
        self.register_buffer("mask", torch.ones(size) if keep is None else keep)

    def factor(self):
        return 1 + self.mask * (self.scale - 1)

    def forward(self, x):
        if self.on_input:
            return self.base(x * self.factor().to(x.dtype))
        out = self.base(x)
        return out * self.factor().to(out.dtype)


def add_ia3(root):
    count = 0
    for module in list(root.modules()):
        qkv = getattr(module, "qkv", None)
        if isinstance(qkv, nn.Linear):
            width = qkv.out_features // 3
            keep = torch.zeros(qkv.out_features)
            keep[width:] = 1
            module.qkv = IA3Linear(qkv, qkv.out_features, False, keep)
            count += 1
        for name in ("k", "v"):
            layer = getattr(module, name, None)
            if isinstance(layer, nn.Linear):
                setattr(module, name, IA3Linear(layer, layer.out_features, False))
                count += 1
        down = getattr(module, "down", None)
        if isinstance(down, nn.Linear):
            module.down = IA3Linear(down, down.in_features, True)
            count += 1
    return count


class RoleCRF(nn.Module):
    """Linear-chain CRF over sentence roles. Any role may follow any role."""

    def __init__(self, n_labels):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(n_labels))
        self.end = nn.Parameter(torch.zeros(n_labels))
        self.trans = nn.Parameter(torch.zeros(n_labels, n_labels))

    def nll(self, emissions, tags):
        steps = emissions.size(0)
        gold = self.start[tags[0]] + emissions[0, tags[0]] + self.end[tags[-1]]
        if steps > 1:
            gold = gold + self.trans[tags[:-1], tags[1:]].sum()
            gold = gold + emissions[1:].gather(1, tags[1:].unsqueeze(1)).sum()
        alpha = self.start + emissions[0]
        for step in range(1, steps):
            alpha = torch.logsumexp(alpha.unsqueeze(1) + self.trans + emissions[step].unsqueeze(0), dim=0)
        logz = torch.logsumexp(alpha + self.end, dim=0)
        return (logz - gold) / steps

    @torch.no_grad()
    def decode(self, emissions):
        steps = emissions.size(0)
        score = self.start + emissions[0]
        back = []
        for step in range(1, steps):
            best, index = (score.unsqueeze(1) + self.trans).max(dim=0)
            score = best + emissions[step]
            back.append(index)
        last = int((score + self.end).argmax())
        path = [last]
        for index in reversed(back):
            last = int(index[last])
            path.append(last)
        return path[::-1]


DOMAIN_INDEX = {"CL": 0, "IT": 1}


class RoleTagger(nn.Module):
    def __init__(self, backbone, n_labels, hidden=256, layers=1, domain_dim=0, attn_heads=0, buckets=20):
        super().__init__()
        self.backbone = backbone
        width = backbone.embed.weight.size(1)
        self.buckets = buckets
        self.domain_dim = domain_dim
        self.attn_heads = attn_heads
        self.position = nn.Embedding(buckets, 32)
        self.domain = nn.Embedding(2, domain_dim) if domain_dim else None
        self.drop = nn.Dropout(0.2)
        input_size = width + 32 + domain_dim
        self.norm = nn.LayerNorm(input_size)
        self.context = nn.LSTM(
            input_size, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1 if layers > 1 else 0.0
        )
        context_size = 2 * hidden
        if attn_heads:
            self.attn = nn.MultiheadAttention(context_size, attn_heads, dropout=0.1, batch_first=True)
            self.attn_norm = nn.LayerNorm(context_size)
        else:
            self.attn = None
        self.head = nn.Linear(context_size, n_labels)
        self.crf = RoleCRF(n_labels)

    def encode(self, token_ids, pad_mask):
        hidden = hidden_states(self.backbone, token_ids, pad_mask)
        weight = pad_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)

    def emissions(self, vectors, positions, domain="CL"):
        """vectors [sentences, width], positions [sentences] in [0, 1)."""
        bucket = (positions * self.buckets).long().clamp(0, self.buckets - 1)
        parts = [vectors.float(), self.position(bucket)]
        if self.domain is not None:
            index = torch.full((vectors.size(0),), DOMAIN_INDEX[domain], device=vectors.device, dtype=torch.long)
            parts.append(self.domain(index))
        features = self.norm(torch.cat(parts, dim=-1)).unsqueeze(0)
        context, _state = self.context(self.drop(features))
        if self.attn is not None:
            attended, _weights = self.attn(context, context, context, need_weights=False)
            context = self.attn_norm(context + self.drop(attended))
        return self.head(self.drop(context))[0]


def build_tagger(arch, method, weights, rank, alpha, n_labels, head_hidden=256, head_layers=1, domain_dim=0, attn_heads=0):
    backbone = build_model(arch, vocab_size=8192)
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    enable_full_attention(backbone)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    # The encoder-decoder tags with its encoder only; the decoder never runs.
    blocks = backbone.encoder if arch == "encdec" else backbone.blocks
    if method in ("lorma", "lora", "rslora"):
        wrapped = adapt_linears(blocks, "lorma" if method == "lorma" else "lora", rank, alpha)
        if method == "rslora":
            for module in blocks.modules():
                if isinstance(module, LoRALinear):
                    module.scale = alpha / math.sqrt(rank)
        print(f"wrapped {wrapped} linear layers with {method}", flush=True)
    elif method == "ia3":
        print(f"wrapped {add_ia3(blocks)} linear layers with ia3", flush=True)
    elif method != "frozen":
        raise SystemExit(f"unknown method {method}")
    tagger = RoleTagger(backbone, n_labels, head_hidden, head_layers, domain_dim, attn_heads)
    tagger.adapter_keys = {
        name for name, parameter in tagger.named_parameters() if name.startswith("backbone.") and parameter.requires_grad
    }
    print(
        f"head hidden {head_hidden} layers {head_layers} domain_dim {domain_dim} attn_heads {attn_heads}",
        flush=True,
    )
    return tagger


def head_options(checkpoint_or_args):
    """Read head settings from a checkpoint dict or argparse namespace."""
    get = checkpoint_or_args.get if isinstance(checkpoint_or_args, dict) else lambda key, default=None: getattr(
        checkpoint_or_args, key, default
    )
    return {
        "head_hidden": int(get("head_hidden", 256) or 256),
        "head_layers": int(get("head_layers", 1) or 1),
        "domain_dim": int(get("domain_dim", 0) or 0),
        "attn_heads": int(get("attn_heads", 0) or 0),
    }


def backbone_mode(tagger, method, training):
    """Train mode turns on block checkpointing; dropout inside the Phase 1 model stays off."""
    if not training or method == "frozen":
        tagger.backbone.eval()
        return
    tagger.backbone.train()
    for module in tagger.backbone.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def pad_batch(rows, device):
    width = max(len(row) for row in rows)
    token_ids = torch.zeros(len(rows), width, dtype=torch.long)
    pad_mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for index, row in enumerate(rows):
        token_ids[index, : len(row)] = torch.tensor(row)
        pad_mask[index, : len(row)] = True
    return token_ids.to(device), pad_mask.to(device)


def encode_sentences(tagger, rows, device, token_budget=8192):
    """Sentence vectors without gradients, batched by length."""
    order = sorted(range(len(rows)), key=lambda index: len(rows[index]))
    vectors = [None] * len(rows)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and (stop - start + 1) * len(rows[order[stop]]) <= token_budget:
            stop += 1
        chunk = order[start:stop]
        token_ids, pad_mask = pad_batch([rows[index] for index in chunk], device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pooled = tagger.encode(token_ids, pad_mask).float().cpu()
        for slot, index in enumerate(chunk):
            vectors[index] = pooled[slot]
        start = stop
    return torch.stack(vectors)


def positions_of(count, start=0, length=None):
    length = length if length is not None else count
    return torch.arange(start, start + length, dtype=torch.float) / max(1, count)


def doc_vectors(tagger, docs, device):
    tagger.backbone.eval()
    return {doc["id"]: encode_sentences(tagger, doc["ids"], device) for doc in docs}


def doc_emissions(tagger, docs, vectors, device):
    tagger.eval()
    out = {}
    with torch.no_grad():
        for doc in docs:
            sentences = vectors[doc["id"]].to(device)
            out[doc["id"]] = tagger.emissions(
                sentences, positions_of(len(doc["ids"])).to(device), doc["domain"]
            ).float()
    return out


def predict(tagger, docs, vectors, vocab, device):
    emissions = doc_emissions(tagger, docs, vectors, device)
    return {doc_id: [vocab[index] for index in tagger.crf.decode(scores)] for doc_id, scores in emissions.items()}


def macro_f1(docs, predictions, vocab):
    """Same rule as the course evaluator: all 13 roles, zero when undefined."""
    stats = {label: [0, 0, 0] for label in vocab}
    for doc in docs:
        for gold, guess in zip(doc["labels"], predictions[doc["id"]]):
            if gold == guess:
                stats[gold][0] += 1
            else:
                stats[guess][1] += 1
                stats[gold][2] += 1
    lines = []
    f1s = []
    for label in vocab:
        true_pos, false_pos, false_neg = stats[label]
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        lines.append(f"  {label:24s} P={precision:.3f} R={recall:.3f} F1={f1:.3f} gold={true_pos + false_neg}")
    return sum(f1s) / len(f1s), lines


def domain_scores(docs, predictions, vocab):
    parts = []
    for domain in ("CL", "IT"):
        subset = [doc for doc in docs if doc["domain"] == domain]
        if subset:
            parts.append(f"{domain} {macro_f1(subset, predictions, vocab)[0]:.4f}")
    return "  ".join(parts)


def class_weights(docs, vocab, cap):
    counts = torch.ones(len(vocab))
    index = {label: position for position, label in enumerate(vocab)}
    for doc in docs:
        for label in doc["labels"]:
            counts[index[label]] += 1
    weights = 1.0 / counts.sqrt()
    weights = weights * (len(vocab) / weights.sum())
    return torch.clamp(weights, max=float(weights.median() * cap))


def role_loss(tagger, emissions, tags, weights, ce_weight):
    loss = tagger.crf.nll(emissions.float(), tags)
    if ce_weight > 0:
        loss = loss + ce_weight * F.cross_entropy(emissions.float(), tags, weight=weights)
    return loss


def distill_loss(student, teacher, temperature):
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    teacher_prob = F.softmax(teacher.float() / temperature, dim=-1)
    return (temperature * temperature) * F.kl_div(student_log, teacher_prob, reduction="batchmean")


def training_chunks(docs, chunk):
    """Consecutive sentence chunks; the cut points shift every epoch."""
    pieces = []
    for doc in docs:
        count = len(doc["ids"])
        offset = random.randrange(chunk) if count > chunk else 0
        cuts = sorted({0, count, *range(offset, count, chunk)})
        for start, stop in zip(cuts, cuts[1:]):
            pieces.append((doc, start, stop))
    random.shuffle(pieces)
    return pieces


def trained_state(tagger):
    """Only what training changed: the head, the adapters, and their masks."""
    return {
        name: value
        for name, value in tagger.state_dict().items()
        if not name.startswith("backbone.") or name in tagger.adapter_keys or name.endswith(".mask")
    }


def load_trained(tagger, state):
    result = tagger.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise SystemExit(f"checkpoint keys not in the model: {result.unexpected_keys[:5]}")
    missing = [name for name in result.missing_keys if not name.startswith("backbone.")]
    if missing:
        raise SystemExit(f"checkpoint is missing head keys: {missing[:5]}")


def save_checkpoint(path, tagger, args, method, vocab, dev_f1):
    torch.save(
        {
            "arch": args.arch,
            "method": method,
            "rank": args.rank,
            "alpha": args.alpha,
            "weights": args.weights,
            "vocab": vocab,
            "dev_f1": dev_f1,
            "head_hidden": args.head_hidden,
            "head_layers": args.head_layers,
            "domain_dim": args.domain_dim,
            "attn_heads": args.attn_heads,
            "model": trained_state(tagger),
        },
        path,
    )
    print("saved", path, f"dev macro-F1 {dev_f1:.4f}", flush=True)


def teacher_scores(args, docs, n_labels, device):
    """Role scores of the trained teacher for every training sentence."""
    checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
    weights = args.teacher_weights or checkpoint["weights"]
    print("teacher", checkpoint["arch"], checkpoint["method"], "dev", round(checkpoint["dev_f1"], 4), flush=True)
    teacher = build_tagger(
        checkpoint["arch"],
        checkpoint["method"],
        weights,
        checkpoint["rank"],
        checkpoint["alpha"],
        n_labels,
        **head_options(checkpoint),
    )
    load_trained(teacher, checkpoint["model"])
    teacher.to(device)
    scores = doc_emissions(teacher, docs, doc_vectors(teacher, docs, device), device)
    scores = {doc_id: value.cpu() for doc_id, value in scores.items()}
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def write_outputs(out_dir, predictions):
    pred_path = os.path.join(out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False)
    zip_path = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("zip", zip_path, "docs", len(predictions), flush=True)


def write_dev_files(out_dir, dev_docs, predictions):
    """Files for a cross-check with evaluate_phase3.py."""
    for domain in ("CL", "IT"):
        truth = {doc["id"]: doc["labels"] for doc in dev_docs if doc["domain"] == domain}
        with open(os.path.join(out_dir, f"dev_truth_{domain.lower()}.json"), "w", encoding="utf-8") as handle:
            json.dump(truth, handle, ensure_ascii=False)
    with open(os.path.join(out_dir, "dev_prediction.json"), "w", encoding="utf-8") as handle:
        json.dump({doc["id"]: predictions[doc["id"]] for doc in dev_docs}, handle, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("efficient", "decoder", "hybrid", "encdec"), default="efficient")
    parser.add_argument("--method", choices=("frozen", "lorma", "lora", "rslora", "ia3", "distill"), default="lorma")
    parser.add_argument("--student_method", choices=("lorma", "lora", "rslora", "ia3"), default="lora")
    parser.add_argument("--teacher", default=None, help="best.pt of a finished Phase 3 run.")
    parser.add_argument("--teacher_weights", default=None, help="Phase 1 weights of the teacher, if moved.")
    parser.add_argument("--kd_weight", type=float, default=0.5, help="Share of the loss taken by the teacher.")
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=0, help="0 picks 40 for frozen and 8 otherwise.")
    parser.add_argument("--patience", type=int, default=0, help="0 picks 8 for frozen and 3 otherwise.")
    parser.add_argument("--lr", type=float, default=0.0, help="Adapter learning rate; 0 picks one per method.")
    parser.add_argument("--head_lr", type=float, default=1e-3, help="BiLSTM, head, and CRF.")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--chunk", type=int, default=32, help="Consecutive sentences per training step.")
    parser.add_argument("--train_tokens", type=int, default=96, help="Tokens kept per sentence while training.")
    parser.add_argument("--eval_tokens", type=int, default=160, help="Tokens kept per sentence at prediction.")
    parser.add_argument("--weight_cap", type=float, default=5.0)
    parser.add_argument("--ce_weight", type=float, default=1.0, help="Class-weighted cross-entropy added to the CRF loss.")
    parser.add_argument("--max_steps", type=int, default=0, help="Cap steps per epoch, for a quick check.")
    parser.add_argument(
        "--refit_from",
        default=None,
        help="best.pt of a finished run. Its adapters stay fixed; the BiLSTM and CRF retrain on whole judgments.",
    )
    parser.add_argument(
        "--adapters_from",
        default=None,
        help="best.pt whose LoRMA/IA3 keys are loaded into a freshly built head (CLI head_* flags apply).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed", action="store_true", help="Train all epochs and keep the last one, no dev selection.")
    parser.add_argument(
        "--include_dev", action="store_true", help="Also train on the labelled dev documents. Needs --fixed."
    )
    parser.add_argument("--head_hidden", type=int, default=256, help="BiLSTM hidden size per direction.")
    parser.add_argument("--head_layers", type=int, default=1, help="Number of BiLSTM layers.")
    parser.add_argument("--domain_dim", type=int, default=0, help="CL/IT embedding size; 0 turns it off.")
    parser.add_argument("--attn_heads", type=int, default=0, help="Sentence self-attention heads; 0 turns it off.")
    args = parser.parse_args()
    if args.include_dev and not args.fixed:
        raise SystemExit("--include_dev needs --fixed, because the dev score can no longer pick the epoch")
    if args.attn_heads and (2 * args.head_hidden) % args.attn_heads:
        raise SystemExit("2 * head_hidden must divide evenly by attn_heads")

    if args.refit_from and args.adapters_from:
        raise SystemExit("use only one of --refit_from and --adapters_from")
    refit = None
    adapters = None
    if args.refit_from:
        refit = torch.load(args.refit_from, map_location="cpu", weights_only=False)
        args.arch, args.rank, args.alpha = refit["arch"], refit["rank"], refit["alpha"]
        for key, value in head_options(refit).items():
            setattr(args, key, value)
        method = refit["method"]
        if not args.weights:
            args.weights = refit["weights"] if os.path.isfile(refit["weights"]) else DEFAULT_WEIGHTS[args.arch]
        print("refit from", args.refit_from, "dev", round(refit["dev_f1"], 4), flush=True)
    elif args.adapters_from:
        adapters = torch.load(args.adapters_from, map_location="cpu", weights_only=False)
        args.arch, args.rank, args.alpha = adapters["arch"], adapters["rank"], adapters["alpha"]
        method = adapters["method"]
        if not args.weights:
            args.weights = adapters["weights"] if os.path.isfile(adapters["weights"]) else DEFAULT_WEIGHTS[args.arch]
        print(
            "adapters from",
            args.adapters_from,
            "dev",
            round(adapters["dev_f1"], 4),
            "new head",
            args.head_hidden,
            args.head_layers,
            args.domain_dim,
            args.attn_heads,
            flush=True,
        )
    else:
        method = args.student_method if args.method == "distill" else args.method
    if args.method == "distill" and not args.teacher:
        raise SystemExit("distill needs --teacher pointing at a finished Phase 3 best.pt")
    cached_mode = method == "frozen" or refit is not None or adapters is not None
    args.weights = args.weights or DEFAULT_WEIGHTS[args.arch]
    args.lr = args.lr or DEFAULT_LR[method]
    if (refit is not None or adapters is not None) and args.head_lr == 1e-3:
        args.head_lr = 5e-4
    epochs = args.epochs or (40 if cached_mode else 8)
    patience = args.patience or (8 if cached_mode else 3)

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "arch", args.arch, "method", args.method, "adapter", method, "seed", args.seed, flush=True)
    print("weights", args.weights, "lr", args.lr, "head_lr", args.head_lr, flush=True)

    vocab = read_vocab(args.data_dir)
    label_index = {label: position for position, label in enumerate(vocab)}
    train_docs = load_docs(args.data_dir, TRAIN_FILES)
    dev_docs = load_docs(args.data_dir, DEV_FILES)
    test_docs = load_docs(args.data_dir, TEST_FILES)
    every_doc = train_docs + dev_docs + test_docs
    if args.include_dev:
        train_docs = train_docs + dev_docs
    print("docs train", len(train_docs), "dev", len(dev_docs), "test", len(test_docs), flush=True)

    tokenizer = LegalBPE.load(args.bpe)
    tokenize_docs(tokenizer, every_doc, args.eval_tokens)

    teacher = None
    if args.teacher:
        teacher = teacher_scores(args, train_docs, len(vocab), device)

    tagger = build_tagger(
        args.arch,
        method,
        args.weights,
        args.rank,
        args.alpha,
        len(vocab),
        args.head_hidden,
        args.head_layers,
        args.domain_dim,
        args.attn_heads,
    ).to(device)
    if refit is not None:
        load_trained(tagger, refit["model"])
        for parameter in tagger.backbone.parameters():
            parameter.requires_grad_(False)
    elif adapters is not None:
        adapter_state = {
            name: value
            for name, value in adapters["model"].items()
            if name in tagger.adapter_keys or name.endswith(".mask")
        }
        missing = tagger.adapter_keys - set(adapter_state)
        if missing:
            raise SystemExit(f"adapters_from is missing keys: {list(missing)[:5]}")
        tagger.load_state_dict(adapter_state, strict=False)
        for parameter in tagger.backbone.parameters():
            parameter.requires_grad_(False)
    weights = class_weights(train_docs, vocab, args.weight_cap).to(device)
    print("class weights", {label: round(float(value), 2) for label, value in zip(vocab, weights)}, flush=True)

    head_params = [p for name, p in tagger.named_parameters() if not name.startswith("backbone.")]
    adapter_params = [p for name, p in tagger.named_parameters() if name.startswith("backbone.") and p.requires_grad]
    groups = [{"params": head_params, "lr": args.head_lr}]
    if adapter_params:
        groups.append({"params": adapter_params, "lr": args.lr})
    backbone_total = sum(p.numel() for p in tagger.backbone.parameters())
    print(
        f"Phase 1 parameters {backbone_total / 1e6:.2f}M  trainable adapter {sum(p.numel() for p in adapter_params) / 1e6:.3f}M  "
        f"head {sum(p.numel() for p in head_params) / 1e6:.2f}M",
        flush=True,
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_path = os.path.join(args.out, "best.pt")
    all_params = [p for group in groups for p in group["params"]]

    cached = None
    if cached_mode:
        started = time.time()
        cached = doc_vectors(tagger, every_doc, device)
        print(f"cached sentence vectors {time.time() - started:.0f}s", flush=True)

    best_f1 = -1.0
    if refit is not None or adapters is not None:
        dev_pred = predict(tagger, dev_docs, cached, vocab, device)
        best_f1 = macro_f1(dev_docs, dev_pred, vocab)[0]
        print(f"epoch 0 dev macro-F1 {best_f1:.4f}  ({domain_scores(dev_docs, dev_pred, vocab)})", flush=True)
        if refit is not None:
            save_checkpoint(best_path, tagger, args, method, vocab, best_f1)
    waited = 0
    run_started = time.time()
    for epoch in range(1, epochs + 1):
        started = time.time()
        tagger.train()
        backbone_mode(tagger, "frozen" if cached_mode else method, True)
        total = 0.0
        steps = 0
        if cached_mode:
            pieces = [(doc, 0, len(doc["ids"])) for doc in train_docs]
            random.shuffle(pieces)
        else:
            pieces = training_chunks(train_docs, args.chunk)
        for doc, start, stop in pieces:
            tags = torch.tensor([label_index[label] for label in doc["labels"][start:stop]], device=device)
            positions = positions_of(len(doc["ids"]), start, stop - start).to(device)
            if cached_mode:
                vectors = cached[doc["id"]][start:stop].to(device)
            else:
                rows = [ids[: args.train_tokens] for ids in doc["ids"][start:stop]]
                token_ids, pad_mask = pad_batch(rows, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    vectors = tagger.encode(token_ids, pad_mask)
            emissions = tagger.emissions(vectors.float(), positions, doc["domain"])
            loss = role_loss(tagger, emissions, tags, weights, args.ce_weight)
            if teacher is not None:
                soft = teacher[doc["id"]][start:stop].to(device)
                loss = (1 - args.kd_weight) * loss + args.kd_weight * distill_loss(emissions, soft, args.kd_temp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            steps += 1
            if not cached_mode and steps % 100 == 0:
                print(f"  step {steps}/{len(pieces)} loss {total / steps:.4f} {time.time() - started:.0f}s", flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        print(f"epoch {epoch} train loss {total / max(1, steps):.4f} {time.time() - started:.0f}s", flush=True)

        vectors = cached if cached_mode else doc_vectors(tagger, dev_docs, device)
        dev_pred = predict(tagger, dev_docs, vectors, vocab, device)
        f1, lines = macro_f1(dev_docs, dev_pred, vocab)
        print(f"epoch {epoch} dev macro-F1 {f1:.4f}  ({domain_scores(dev_docs, dev_pred, vocab)})", flush=True)
        if args.fixed:
            if epoch == epochs:
                save_checkpoint(best_path, tagger, args, method, vocab, f1)
        elif f1 > best_f1:
            best_f1 = f1
            waited = 0
            save_checkpoint(best_path, tagger, args, method, vocab, f1)
        else:
            waited += 1
            if waited >= patience:
                print("early stop", flush=True)
                break
    print(f"training time {(time.time() - run_started) / 60:.1f} min", flush=True)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    load_trained(tagger, checkpoint["model"])
    tagger.to(device)
    if not cached_mode:
        cached = doc_vectors(tagger, dev_docs + test_docs, device)
    dev_pred = predict(tagger, dev_docs, cached, vocab, device)
    f1, lines = macro_f1(dev_docs, dev_pred, vocab)
    print(f"best checkpoint dev macro-F1 {f1:.4f}  ({domain_scores(dev_docs, dev_pred, vocab)})", flush=True)
    print("\n".join(lines), flush=True)
    write_dev_files(args.out, dev_docs, dev_pred)
    test_pred = predict(tagger, test_docs, cached, vocab, device)
    for doc in test_docs:
        if len(test_pred[doc["id"]]) != len(doc["text"]):
            raise SystemExit(f"{doc['id']}: {len(test_pred[doc['id']])} roles for {len(doc['text'])} sentences")
    write_outputs(args.out, test_pred)


if __name__ == "__main__":
    main()
