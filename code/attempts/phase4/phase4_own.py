"""Phase 4 court-judgment prediction from the Phase 1 models.

Each case is read as a few token windows spread through the document.
The Phase 1 model stays frozen except for the chosen adaptation:
LoRMA, BitFit, a bottleneck adapter, the last blocks, or distillation.

    python phase4_own.py --arch efficient --method lorma --out runs/eff_lorma
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
USE_FP16 = True
DEFAULT_WEIGHTS = {
    "efficient": os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\model.pt"),
    "decoder": os.path.join(LOCAL, r"local_weights\cs781_phase1_decoder\model.pt"),
    "hybrid": os.path.join(LOCAL, r"local_weights\cs781_phase1_hybrid\model.pt"),
    "encdec": os.path.join(LOCAL, r"encdec_run2\cs781_phase1_encdec\model.pt"),
}
DEFAULT_BPE = os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\legal_bpe.json")
DEFAULT_DATA = os.path.join(LOCAL, "phase4data")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path, labeled):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for item in payload:
        text = item.get("text") or ""
        label = item.get("label") if labeled else None
        if labeled and label not in (0, 1):
            continue
        if not str(text).strip():
            continue
        rows.append({"id": str(item["id"]), "text": text, "label": None if label is None else int(label)})
    return rows


def shorten(text, word_cap, focus="ends"):
    words = text.split()
    if len(words) <= word_cap:
        return text
    if focus == "tail":
        return " ".join(words[-word_cap:])
    head = word_cap * 2 // 3
    return " ".join(words[:head] + words[-(word_cap - head) :])


def windows_of(ids, count, width, cover="spread"):
    if not ids:
        return [[1]]
    if cover == "tail":
        ids = ids[-(count * width) :]
        parts = [ids[start : start + width] for start in range(0, len(ids), width)]
        return parts or [[1]]
    if len(ids) <= width:
        return [ids]
    if len(ids) <= count * width:
        return [ids[start : start + width] for start in range(0, len(ids), width)][:count]
    span = len(ids) - width
    starts = [int(round(i * span / (count - 1))) for i in range(count)]
    return [ids[start : start + width] for start in starts]


class BlockAdapter(nn.Module):
    """Frozen block plus a bottleneck residual, starting at zero."""

    def __init__(self, block, width, rank):
        super().__init__()
        self.block = block
        for parameter in block.parameters():
            parameter.requires_grad_(False)
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, pad_mask=None):
        hidden = self.block(x, pad_mask=pad_mask)
        return hidden + self.up(F.relu(self.down(hidden)))


def add_adapters(blocks, width, rank):
    for index, block in enumerate(list(blocks)):
        blocks[index] = BlockAdapter(block, width, rank)
    return len(blocks)


def apply_method(model, arch, method, rank, alpha, unfreeze=2):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = model.encoder if arch == "encdec" else model.blocks
    if method in ("lorma", "lora"):
        wrapped = adapt_linears(blocks, "lorma" if method == "lorma" else "lora", rank, alpha)
        print(f"wrapped {wrapped} linear layers with {method}", flush=True)
    elif method == "adapter":
        width = model.embed.weight.size(1)
        print(f"added adapters on {add_adapters(blocks, width, rank)} blocks", flush=True)
    elif method == "bitfit":
        count = 0
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.requires_grad_(True)
                count += parameter.numel()
        print(f"bitfit bias parameters {count}", flush=True)
    elif method == "partial":
        chosen = list(blocks)[-max(1, unfreeze) :]
        count = 0
        for block in chosen:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
                count += parameter.numel()
        print(f"unfroze last {len(chosen)} blocks, {count} parameters", flush=True)
    else:
        raise SystemExit(f"unknown method {method}")


class CaseClassifier(nn.Module):
    def __init__(self, backbone, arch, extra=0):
        super().__init__()
        self.backbone = backbone
        self.arch = arch
        self.extra = extra
        width = backbone.embed.weight.size(1)
        self.token_score = nn.Linear(width, 1)
        self.chunk_score = nn.Linear(width, 1)
        self.end_bias = nn.Parameter(torch.tensor(2.0))
        self.head = nn.Linear(width + extra, 2)
        nn.init.zeros_(self.token_score.weight)
        nn.init.zeros_(self.token_score.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if extra:
            with torch.no_grad():
                self.head.weight[0, -1] = -1.0
                self.head.weight[1, -1] = 1.0

    def encode_windows(self, windows, device):
        order = sorted(range(len(windows)), key=lambda index: len(windows[index]))
        vectors = [None] * len(windows)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and (stop - start + 1) * len(windows[order[stop]]) <= 4096:
                stop += 1
            chunk = order[start:stop]
            width = max(len(windows[index]) for index in chunk)
            token_ids = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
            pad_mask = torch.zeros(len(chunk), width, dtype=torch.bool, device=device)
            for row, index in enumerate(chunk):
                ids = windows[index]
                token_ids[row, : len(ids)] = torch.tensor(ids, device=device)
                pad_mask[row, : len(ids)] = True
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_FP16 and device.type == "cuda"):
                hidden = hidden_states(self.backbone, token_ids, pad_mask)
            score = self.token_score(hidden.float()).squeeze(-1)
            positions = torch.arange(width, device=device, dtype=torch.float32) / max(width - 1, 1)
            score = score + self.end_bias * positions.unsqueeze(0)
            score = score.masked_fill(~pad_mask, -1e4)
            weight = torch.softmax(score, dim=-1).unsqueeze(-1)
            pooled = (hidden.float() * weight).sum(dim=1)
            for row, index in enumerate(chunk):
                vectors[index] = pooled[row]
            start = stop
        return torch.stack(vectors)

    def logits_many(self, batch_windows, device, extras=None):
        flat = [window for windows in batch_windows for window in windows]
        vectors = self.encode_windows(flat, device)
        if extras is None:
            extras = torch.zeros(len(batch_windows), self.extra, device=device)
        outputs = []
        start = 0
        for row_index, windows in enumerate(batch_windows):
            stop = start + len(windows)
            picked = vectors[start:stop]
            mix = torch.softmax(self.chunk_score(picked).squeeze(-1), dim=0).unsqueeze(-1)
            pooled = (mix * picked).sum(dim=0)
            if self.extra:
                pooled = torch.cat([pooled, extras[row_index].to(pooled.device)], dim=0)
            outputs.append(self.head(pooled))
            start = stop
        return torch.stack(outputs)

    def logits(self, windows, device):
        return self.logits_many([windows], device)[0]


def macro_f1(gold, pred):
    scores = []
    for label in (0, 1):
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / 2


def sentence_windows(text, count, width, tokenizer):
    parts = []
    for piece in str(text).replace("!", ".").replace("?", ".").split("."):
        if len(piece.split()) >= 4:
            parts.append(piece.strip())
    windows = []
    for part in parts[-count:]:
        ids = tokenizer.encode(part)
        if len(ids) > width:
            ids = ids[-width:]
        if ids:
            windows.append(ids)
    return windows or [[1]]


def prepare(rows, tokenizer, chunks, width, word_cap, focus="ends", cover="spread", tail_tokens=0, sentences=0):
    prepared = []
    for row in rows:
        if sentences:
            windows = sentence_windows(row["text"], sentences, width, tokenizer)
        else:
            ids = tokenizer.encode(shorten(row["text"], word_cap, focus))
            if tail_tokens:
                windows = [ids[-tail_tokens:] or [1]]
            else:
                windows = windows_of(ids, chunks, width, cover)
        item = {
            "id": row["id"],
            "label": row["label"],
            "windows": windows,
        }
        if "word_prob" in row:
            item["word_prob"] = float(row["word_prob"])
        prepared.append(item)
    return prepared


def batch_extras(batch, device):
    if "word_prob" not in batch[0]:
        return None
    return torch.tensor([[row["word_prob"]] for row in batch], dtype=torch.float32, device=device)


def positive_scores(model, rows, device, batch_size=16):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            logits = model.logits_many(
                [row["windows"] for row in batch], device, batch_extras(batch, device)
            )
            scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return scores


def best_cut(gold, positive):
    best_f1, best_threshold = -1.0, 0.5
    for cut in [index / 100 for index in range(15, 86)]:
        pred = [1 if score >= cut else 0 for score in positive]
        f1 = macro_f1(gold, pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, cut
    return best_threshold, best_f1


def balanced_indices(rows):
    groups = {0: [], 1: []}
    for index, row in enumerate(rows):
        groups[int(row["label"])].append(index)
    width = max(len(groups[0]), len(groups[1]), 1)
    order = []
    for label in (0, 1):
        pool = groups[label] or [0]
        picked = [pool[index % len(pool)] for index in range(width)]
        random.shuffle(picked)
        order.extend(picked)
    random.shuffle(order)
    return order


def set_lrs(optimizer, step, total_steps, warmup, bases):
    if total_steps <= 1 or warmup <= 0 and step >= total_steps:
        scale = 1.0
    elif step < warmup:
        scale = (step + 1) / max(1, warmup)
    else:
        progress = (step - warmup) / max(1, total_steps - warmup)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group, base in zip(optimizer.param_groups, bases):
        group["lr"] = base * scale


def dump_scores(path, rows, scores):
    payload = [{"id": row["id"], "score": round(float(score), 6)} for row, score in zip(rows, scores)]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def predict(model, rows, device, batch_size=16):
    model.eval()
    guesses = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            logits = model.logits_many(
                [row["windows"] for row in batch], device, batch_extras(batch, device)
            )
            guesses.extend(logits.argmax(dim=-1).tolist())
    return guesses


def teacher_logits(args, rows, device):
    checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher = build_case(checkpoint["arch"], checkpoint["method"], checkpoint["weights"], checkpoint["rank"], checkpoint["alpha"])
    teacher.load_state_dict(checkpoint["model"], strict=False)
    teacher.to(device)
    teacher.eval()
    print("teacher", checkpoint["arch"], checkpoint["method"], "dev", round(checkpoint["dev_f1"], 4), flush=True)
    scores = []
    with torch.no_grad():
        for row in rows:
            scores.append(teacher.logits(row["windows"], device).float().cpu())
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def build_case(arch, method, weights, rank, alpha, unfreeze=2, extra=0):
    backbone = build_model(arch, vocab_size=8192)
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    enable_full_attention(backbone)
    apply_method(backbone, arch, method, rank, alpha, unfreeze)
    return CaseClassifier(backbone, arch, extra)


def trainable_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if _keep(model, name)}


def _keep(model, name):
    if not name.startswith("backbone."):
        return True
    parameter_name = name.replace(".weight", "").replace(".bias", "")
    # Keep every tensor whose matching parameter is trainable, plus adapter masks.
    named = dict(model.named_parameters())
    if name in named and named[name].requires_grad:
        return True
    return name.endswith(".mask")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("efficient", "decoder", "hybrid", "encdec"), default="efficient")
    parser.add_argument("--method", choices=("lorma", "lora", "bitfit", "adapter", "partial", "distill"), default="lorma")
    parser.add_argument("--student_method", choices=("lorma", "lora", "adapter", "bitfit", "partial"), default="lora")
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_docs", type=int, default=3000)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--chunk_tokens", type=int, default=96)
    parser.add_argument("--word_cap", type=int, default=6000)
    parser.add_argument("--focus", choices=("ends", "tail"), default="ends")
    parser.add_argument("--cover", choices=("spread", "tail"), default="spread")
    parser.add_argument("--unfreeze", type=int, default=2)
    parser.add_argument("--word_feature", action="store_true")
    parser.add_argument("--tail_tokens", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--unfreeze_embed", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--kd_weight", type=float, default=0.5)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--sentences", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--skip_docs", type=int, default=0)
    parser.add_argument("--train_dropout", action="store_true")
    args = parser.parse_args()
    if args.method == "distill" and not args.teacher:
        raise SystemExit("distill needs --teacher")
    method = args.student_method if args.method == "distill" else args.method
    args.weights = args.weights or DEFAULT_WEIGHTS[args.arch]
    if method == "bitfit":
        args.lr = max(args.lr, 1e-3)
    if method == "partial":
        args.lr = min(args.lr, 1e-4)

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "arch", args.arch, "method", args.method, "adapter", method, flush=True)

    root = os.path.join(args.data_dir, "Phase_4_CJPE")
    train_rows = read_rows(os.path.join(root, "Train", "train.json"), True)
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), False)
    print("docs", len(train_rows), len(dev_rows), len(test_rows), flush=True)
    extra = 1 if args.word_feature else 0
    if args.word_feature:
        from phase4_words import fit_word_model, word_probabilities

        saved = os.path.join(LOCAL, "phase4_runs", "word_end.joblib")
        word_path = saved if os.path.isfile(saved) else os.path.join(args.out, "word_end.joblib")
        bundle = fit_word_model(train_rows, word_path)
        for split in (train_rows, dev_rows, test_rows):
            probs = word_probabilities(bundle, [row["text"] for row in split])
            for row, prob in zip(split, probs):
                row["word_prob"] = float(prob)
        print("word feature on", flush=True)
    random.shuffle(train_rows)
    if args.skip_docs:
        train_rows = train_rows[args.skip_docs :]
        print("skipped", args.skip_docs, "already used docs", flush=True)
    train_rows = train_rows[: args.max_docs]
    counts = [sum(row["label"] == label for row in train_rows) for label in (0, 1)]
    if args.balanced:
        weights = torch.ones(2, device=device)
    else:
        weights = torch.tensor([len(train_rows) / (2 * max(1, count)) for count in counts], device=device)
    print("train subset", len(train_rows), "class weights", [round(float(value), 3) for value in weights], flush=True)

    tokenizer = LegalBPE.load(args.bpe)
    started = time.time()
    train_rows = prepare(
        train_rows, tokenizer, args.chunks, args.chunk_tokens, args.word_cap, args.focus, args.cover, args.tail_tokens, args.sentences
    )
    dev_rows = prepare(
        dev_rows, tokenizer, args.chunks, args.chunk_tokens, args.word_cap, args.focus, args.cover, args.tail_tokens, args.sentences
    )
    test_rows = prepare(
        test_rows, tokenizer, args.chunks, args.chunk_tokens, args.word_cap, args.focus, args.cover, args.tail_tokens, args.sentences
    )
    print(
        "focus", args.focus, "cover", args.cover, "chunks", args.chunks,
        "width", args.chunk_tokens, "tail", args.tail_tokens, flush=True,
    )
    print(f"tokenized {time.time() - started:.0f}s", flush=True)

    model = build_case(args.arch, method, args.weights, args.rank, args.alpha, args.unfreeze, extra).to(device)
    if args.unfreeze_embed:
        model.backbone.embed.weight.requires_grad_(True)
        print("unfroze token embeddings", flush=True)
    soft = teacher_logits(args, train_rows, device) if args.method == "distill" else None

    head = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    body = [parameter for name, parameter in model.named_parameters() if name.startswith("backbone.") and parameter.requires_grad]
    groups = [{"params": head, "lr": args.head_lr}]
    if body:
        groups.append({"params": body, "lr": args.lr})
    print(
        f"trainable head {sum(p.numel() for p in head)/1e6:.3f}M  body {sum(p.numel() for p in body)/1e6:.3f}M",
        flush=True,
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    best_f1 = -1.0
    best_path = os.path.join(args.out, "best.pt")
    if args.resume:
        prior = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(prior["model"], strict=False)
        best_f1 = float(prior.get("dev_f1", -1.0))
        torch.save(prior, best_path)
        print(
            f"resumed dev {best_f1:.4f} missing {len(missing)} unexpected {len(unexpected)}",
            flush=True,
        )
    steps_per_epoch = math.ceil(len(train_rows) / max(1, args.batch))
    if args.balanced:
        steps_per_epoch = math.ceil((2 * max(counts)) / max(1, args.batch))
    total_steps = max(1, steps_per_epoch * args.epochs)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        if not args.train_dropout:
            for module in model.backbone.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
        order = balanced_indices(train_rows) if args.balanced else list(range(len(train_rows)))
        if not args.balanced:
            random.shuffle(order)
        total = 0.0
        recent = 0.0
        step = 0
        batch_size = args.batch
        for start in range(0, len(order), batch_size):
            step += 1
            global_step = (epoch - 1) * steps_per_epoch + step
            set_lrs(optimizer, global_step, total_steps, args.warmup, base_lrs)
            chosen = order[start : start + batch_size]
            batch = [train_rows[index] for index in chosen]
            logits = model.logits_many(
                [row["windows"] for row in batch], device, batch_extras(batch, device)
            )
            target = torch.tensor([row["label"] for row in batch], device=device)
            loss = F.cross_entropy(logits, target, weight=weights, label_smoothing=args.label_smoothing)
            if soft is not None:
                teacher = torch.stack([soft[index] for index in chosen]).to(device)
                log_p = F.log_softmax(logits / args.kd_temp, dim=-1)
                q = F.softmax(teacher / args.kd_temp, dim=-1)
                loss = (1 - args.kd_weight) * loss + args.kd_weight * (args.kd_temp ** 2) * F.kl_div(
                    log_p, q, reduction="batchmean"
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(head + body, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            recent += loss.item()
            if step == 1:
                print(f"  grad norm {float(grad_norm):.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
            if step % 50 == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(order)} loss {recent / 50:.4f} {time.time() - started:.0f}s",
                    flush=True,
                )
                recent = 0.0
            if args.eval_every and step % args.eval_every == 0:
                dev_scores = positive_scores(model, dev_rows, device)
                gold = [row["label"] for row in dev_rows]
                step_f1 = macro_f1(gold, [1 if score >= 0.5 else 0 for score in dev_scores])
                print(f"  dev argmax {step_f1:.4f} at step {step}", flush=True)
                if step_f1 > best_f1:
                    best_f1 = step_f1
                    torch.save(
                        {
                            "arch": args.arch,
                            "method": method,
                            "rank": args.rank,
                            "alpha": args.alpha,
                            "weights": args.weights,
                            "dev_f1": step_f1,
                            "model": trainable_state(model),
                        },
                        best_path,
                    )
                    print("saved", best_path, flush=True)
                model.train()
                for module in model.backbone.modules():
                    if isinstance(module, nn.Dropout):
                        module.eval()
        dev_scores = positive_scores(model, dev_rows, device)
        gold = [row["label"] for row in dev_rows]
        argmax_f1 = macro_f1(gold, [1 if score >= 0.5 else 0 for score in dev_scores])
        _, tuned_f1 = best_cut(gold, dev_scores)
        print(
            f"epoch {epoch} loss {total / max(1, step):.4f} dev argmax {argmax_f1:.4f} tuned {tuned_f1:.4f} {time.time() - started:.0f}s",
            flush=True,
        )
        if argmax_f1 > best_f1:
            best_f1 = argmax_f1
            torch.save(
                {
                    "arch": args.arch,
                    "method": method,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "weights": args.weights,
                    "dev_f1": argmax_f1,
                    "model": trainable_state(model),
                },
                best_path,
            )
            print("saved", best_path, flush=True)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model = build_case(args.arch, method, args.weights, args.rank, args.alpha, args.unfreeze, extra)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    dev_scores = positive_scores(model, dev_rows, device)
    gold = [row["label"] for row in dev_rows]
    argmax_f1 = macro_f1(gold, [1 if score >= 0.5 else 0 for score in dev_scores])
    cut, tuned_f1 = best_cut(gold, dev_scores)
    # Argmax tracks the hidden test score. A wild dev threshold does not.
    use_cut = cut if tuned_f1 >= argmax_f1 + 0.015 and 0.35 <= cut <= 0.65 else 0.5
    chosen_f1 = tuned_f1 if use_cut != 0.5 else argmax_f1
    print(f"best dev macro-F1 {chosen_f1:.4f} threshold {use_cut:.2f} argmax {argmax_f1:.4f}", flush=True)
    test_scores = positive_scores(model, test_rows, device)
    dump_scores(os.path.join(args.out, "dev_scores.json"), dev_rows, dev_scores)
    dump_scores(os.path.join(args.out, "test_scores.json"), test_rows, test_scores)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"argmax": argmax_f1, "tuned": tuned_f1, "threshold": use_cut, "chosen": chosen_f1},
            handle,
        )
    predictions = [
        {"id": row["id"], "pred": int(score >= use_cut)} for row, score in zip(test_rows, test_scores)
    ]
    pred_path = os.path.join(args.out, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False)
    zip_path = os.path.join(args.out, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("zip", zip_path, "rows", len(predictions), flush=True)


if __name__ == "__main__":
    main()
