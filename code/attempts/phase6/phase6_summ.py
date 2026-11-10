"""Phase 6 abstractive summarization from the four Phase 1 models.

Causal models (decoder, hybrid, efficient) do SFT: document + BOS + summary,
loss only on the summary tokens (Module 3.1). Encoder-decoder encodes the
document and decodes the summary. Do not use EncoderDecoderLM.forward — that
split was for Phase 1 language modeling.

    python phase6_summ.py --arch decoder --method lorma --out ... --smoke --require_cuda
"""
import argparse
import json
import math
import os
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import re

from legal_bpe import BOS_ID, EOS_ID, PAD_ID, LegalBPE
from phase1_models import EncoderDecoderLM, BLOCK, build_model, generate, sample_top_p
from phase2_lner import adapt_linears, hidden_states
from phase4_own import DEFAULT_BPE, DEFAULT_WEIGHTS, LOCAL, add_adapters, set_lrs, set_seed

DEFAULT_DATA = os.path.join(LOCAL, "phase6data")


class CrossAdapter(nn.Module):
    """Houlsby-style bottleneck on an encoder-decoder cross-attention block."""

    def __init__(self, block, width, rank):
        super().__init__()
        self.block = block
        for parameter in block.parameters():
            parameter.requires_grad_(False)
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, memory, pad_mask=None, memory_mask=None):
        hidden = self.block(x, memory, pad_mask=pad_mask, memory_mask=memory_mask)
        return hidden + self.up(F.relu(self.down(hidden)))


def read_rows(path, labeled):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    if isinstance(payload, dict):
        first = next(iter(payload.values())) if payload else None
        if first is not None and not isinstance(first, dict):
            for key, value in payload.items():
                rows.append({"id": str(key), "document": str(value), "summary": ""})
            return [row for row in rows if row["id"] and row["document"].strip()]
        items = list(payload.values())
    else:
        items = payload
    for item in items:
        if not isinstance(item, dict):
            continue
        doc = item.get("document") or item.get("text") or ""
        summary = str(item.get("summary") or "") if labeled else ""
        rows.append({"id": str(item.get("id", "")), "document": str(doc), "summary": summary})
    return [row for row in rows if row["id"] and row["document"].strip()]


def clip_ids(ids, budget):
    if len(ids) <= budget:
        return list(ids)
    head = max(1, budget * 2 // 3)
    tail = budget - head
    return list(ids[:head] + ids[-tail:])


def pack_causal(src, tgt, src_budget, tgt_budget):
    src = clip_ids(src, src_budget)
    tgt = list(tgt[:tgt_budget])
    seq = src + [BOS_ID] + tgt + [EOS_ID]
    labels = [-100] * len(src) + [-100] + tgt + [EOS_ID]
    return seq, labels


def rouge_n(pred, gold):
    pred_t, gold_t = pred.lower().split(), gold.lower().split()
    if not pred_t and not gold_t:
        return 1.0
    if not pred_t or not gold_t:
        return 0.0
    from collections import Counter
    overlap = sum((Counter(pred_t) & Counter(gold_t)).values())
    precision, recall = overlap / len(pred_t), overlap / len(gold_t)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def rouge_l(pred, gold):
    left, right = pred.lower().split(), gold.lower().split()
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    prev = [0] * (len(right) + 1)
    for token in left:
        cur = [0]
        for index, other in enumerate(right, 1):
            cur.append(prev[index - 1] + 1 if token == other else max(prev[index], cur[-1]))
        prev = cur
    overlap = prev[-1]
    precision, recall = overlap / len(left), overlap / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def adapt(model, arch, method, rank, alpha, unfreeze):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if arch == "encdec":
        groups = [model.encoder, model.decoder]
    else:
        groups = [model.blocks]
    if method in ("lorma", "lora"):
        wrapped = 0
        for group in groups:
            wrapped += adapt_linears(group, "lorma" if method == "lorma" else "lora", rank, alpha)
        print(f"wrapped {wrapped} linear layers with {method}", flush=True)
    elif method == "adapter":
        width = model.embed.weight.size(1)
        count = 0
        if arch == "encdec":
            count += add_adapters(model.encoder, width, rank)
            for index, block in enumerate(list(model.decoder)):
                model.decoder[index] = CrossAdapter(block, width, rank)
                count += 1
        else:
            count += add_adapters(model.blocks, width, rank)
        print(f"added adapters on {count} blocks", flush=True)
    elif method == "partial":
        count = 0
        for group in groups:
            chosen = list(group)[-max(1, unfreeze):]
            for block in chosen:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
                    count += parameter.numel()
        print(f"unfroze last blocks, {count} parameters", flush=True)
    else:
        raise SystemExit(f"unknown method {method}")
    model.embed.weight.requires_grad_(False)


def trainable_state(model):
    named = dict(model.named_parameters())
    return {name: value.detach().cpu() for name, value in model.state_dict().items()
            if name in named and named[name].requires_grad}


def pad_batch(rows, key, device, pad=PAD_ID):
    width = max(len(row[key]) for row in rows)
    tensor = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        ids = row[key]
        tensor[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[index, :len(ids)] = True
    return tensor, mask


def causal_loss(model, rows, device):
    tokens, mask = pad_batch(rows, "seq", device)
    labels = torch.full_like(tokens, -100)
    for index, row in enumerate(rows):
        labels[index, :len(row["labels"])] = torch.tensor(row["labels"], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits, _ = model(tokens, pad_mask=mask)
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)


def encdec_loss(model, rows, device):
    src, src_mask = pad_batch(rows, "enc", device)
    tgt, tgt_mask = pad_batch(rows, "dec", device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        memory = model.encode(src, src_mask)
        logits = model.decode(tgt[:, :-1], memory, pad_mask=tgt_mask[:, :-1], memory_mask=src_mask)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD_ID)


def kd_loss(student_logits, teacher_logits, keep, temp):
    student = F.log_softmax(student_logits.float() / temp, dim=-1)
    teacher = F.softmax(teacher_logits.float() / temp, dim=-1)
    token = F.kl_div(student, teacher, reduction="none").sum(-1)
    return (token * keep).sum() / keep.sum().clamp(min=1) * (temp * temp)


def teacher_causal_summary_logits(teacher, seqs, starts, widths, device):
    rows = [{"seq": seq} for seq in seqs]
    tokens, mask = pad_batch(rows, "seq", device)
    with torch.no_grad():
        logits, _ = teacher(tokens, pad_mask=mask)
    pieces = []
    for index, (start, width) in enumerate(zip(starts, widths)):
        pieces.append(logits[index, start:start + width])
    return pieces


def block_ngrams(logits, produced, n=3):
    for index, seq in enumerate(produced):
        if len(seq) < n - 1:
            continue
        prefix = tuple(seq[-(n - 1):])
        banned = {seq[j + n - 1] for j in range(len(seq) - n + 1) if tuple(seq[j:j + n - 1]) == prefix}
        recent = set(seq[-24:])
        for token in banned | recent:
            logits[index, token] = logits[index, token] - (4.0 if token in banned else 0.8)


def collapse_repeats(text):
    words = text.split()
    for span in (12, 8, 5, 3):
        out, index, n = [], 0, len(words)
        while index < n:
            if index + 2 * span <= n and words[index:index + span] == words[index + span:index + 2 * span]:
                out.extend(words[index:index + span])
                index += span
                while index + span <= n and words[index:index + span] == out[-span:]:
                    index += span
                continue
            out.append(words[index])
            index += 1
        words = out
    return " ".join(words)


@torch.no_grad()
def generate_encdec_bos(model, idx, max_new_tokens, temperature, top_p, eos_id=EOS_ID, pad_id=PAD_ID, min_new_tokens=0):
    pad_mask = idx != pad_id
    memory = model.encode(idx, pad_mask)
    dec = idx.new_full((idx.size(0), 1), BOS_ID)
    produced = [[] for _ in range(idx.size(0))]
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        active = ~finished
        if not bool(active.any()):
            break
        if dec.size(1) >= model.block_size:
            dec = dec[:, -model.block_size:]
        logits = model.decode(dec, memory, memory_mask=pad_mask)
        next_logits = logits[:, -1].float()
        block_ngrams(next_logits, produced)
        if min_new_tokens and eos_id is not None:
            for position in range(idx.size(0)):
                if len(produced[position]) < min_new_tokens:
                    next_logits[position, eos_id] = -1e9
        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1)
        else:
            next_id = sample_top_p(next_logits / temperature, top_p)
        next_id = torch.where(active, next_id, torch.full_like(next_id, eos_id or pad_id))
        dec = torch.cat([dec, next_id.unsqueeze(1)], dim=1)
        for position in torch.where(active)[0].tolist():
            produced[position].append(int(next_id[position]))
        if eos_id is not None:
            finished = finished | (active & (next_id == eos_id))
    return produced


def chunk_ids(ids, width, overlap=64, cap=6):
    if len(ids) <= width:
        return [list(ids)]
    step = max(1, width - overlap)
    parts = [list(ids[start:start + width]) for start in range(0, len(ids), step)]
    if len(parts) <= cap:
        return parts
    keep = [parts[0]]
    middle = parts[1:-1]
    stride = max(1, math.ceil(len(middle) / max(1, cap - 2)))
    keep.extend(middle[::stride][: cap - 2])
    keep.append(parts[-1])
    return keep


def split_sents(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.split(r"(?<=[.?;:])\s+(?=[A-Z(\[])", text)
    sents = [part.strip() for part in parts if len(part.split()) >= 6]
    return sents or ([text[:1500]] if text else ["summary"])


def embed_texts(model, tokenizer, texts, device, cap=128):
    vecs = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            ids = [tokenizer.encode(text)[:cap] or [BOS_ID] for text in batch]
            width = max(len(row) for row in ids)
            tensor = torch.full((len(ids), width), PAD_ID, dtype=torch.long, device=device)
            mask = torch.zeros(len(ids), width, dtype=torch.bool, device=device)
            for index, row in enumerate(ids):
                tensor[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
                mask[index, :len(row)] = True
            hidden = hidden_states(model, tensor, mask)
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
            vecs.append(F.normalize(pooled.float(), dim=-1))
    return torch.cat(vecs, 0)


def fill_extract(document, draft, model, tokenizer, device, budget):
    sents = split_sents(document)
    if not sents:
        return " ".join(str(document).split()[:budget]) or "summary"
    n = len(sents)
    vecs = embed_texts(model, tokenizer, sents, device)
    doc_vec = F.normalize(vecs.mean(0, keepdim=True), dim=-1).squeeze(0)
    scores = (vecs @ doc_vec).tolist()
    keys = ("held", "appeal", "section", "article", "accordingly", "accused",
            "assessee", "appellant", "respondent", "dismissed", "allowed")
    for index, sent in enumerate(sents):
        frac = index / max(1, n - 1)
        pos = 0.45 if index < 6 else (0.55 if frac >= 0.75 else 0.05)
        if index < 4:
            pos += 0.2
        lex = rouge_n(sent, draft) if draft else 0.0
        low = sent.lower()
        key = 0.08 * sum(word in low for word in keys)
        scores[index] = 0.4 * scores[index] + pos + 0.35 * lex + key
    ranked = sorted(range(n), key=lambda index: scores[index], reverse=True)
    forced = list(range(min(4, n))) + list(range(max(0, n - 8), n))
    chosen, picked, words = set(), [], 0

    def take(index):
        nonlocal words
        if index < 0 or index >= n or index in chosen:
            return False
        piece = sents[index].split()
        if words and words + len(piece) > budget:
            return False
        chosen.add(index)
        picked.append(index)
        words += len(piece)
        return True

    for index in forced + ranked:
        if words >= budget:
            break
        if take(index):
            take(index - 1)
            take(index + 1)
    picked.sort()
    filled = list(picked)
    for left, right in zip(picked, picked[1:]):
        if right == left + 2:
            take(left + 1)
    filled = sorted(chosen)
    text = " ".join(sents[index] for index in filled).strip()
    return text or " ".join(str(document).split()[:budget])


def decode_new(tokenizer, produced, fallback):
    texts = []
    for ids, source in zip(produced, fallback):
        if EOS_ID in ids:
            ids = ids[:ids.index(EOS_ID)]
        text = collapse_repeats(tokenizer.decode(ids).strip())
        if not text:
            text = tokenizer.decode(source[:48]).strip() or "summary"
        texts.append(text)
    return texts


def generate_rows(model, tokenizer, rows, device, src_budget, max_new, temperature=0.0, gen_batch=4, hier=False, min_new=0, extract_words=0):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), gen_batch):
            chunk = rows[start:start + gen_batch]
            prompts = []
            sources = []
            for row in chunk:
                src = list(row["src"])
                if hier and len(src) > src_budget * 2:
                    pieces = []
                    for part in chunk_ids(src, src_budget):
                        mini = [{"src": part, "document": tokenizer.decode(part)}]
                        pieces.append(generate_rows(model, tokenizer, mini, device, src_budget, max(32, max_new // 3),
                                                    temperature, gen_batch=1, hier=False, min_new=0)[0])
                    src = tokenizer.encode(" ".join(pieces))[:src_budget]
                else:
                    src = clip_ids(src, src_budget)
                sources.append(src)
                prompts.append(src if isinstance(model, EncoderDecoderLM) else src + [BOS_ID])
            if max_new <= 0:
                drafts = [""] * len(chunk)
            else:
                width = max(len(prompt) for prompt in prompts)
                tensor = torch.full((len(prompts), width), PAD_ID, dtype=torch.long, device=device)
                for index, prompt in enumerate(prompts):
                    tensor[index, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)
                if isinstance(model, EncoderDecoderLM):
                    produced = generate_encdec_bos(model, tensor, max_new, temperature, 0.9, min_new_tokens=min_new)
                else:
                    produced = generate(model, tensor, max_new_tokens=max_new, temperature=temperature, top_p=0.9,
                                        eos_id=EOS_ID, pad_id=PAD_ID, min_new_tokens=min_new)
                drafts = decode_new(tokenizer, produced, sources)
            if extract_words:
                drafts = [fill_extract(row.get("document") or tokenizer.decode(row["src"]), draft, model, tokenizer, device, extract_words)
                          for row, draft in zip(chunk, drafts)]
            out.extend(drafts)
    return out


def score_rows(preds, golds):
    r1 = float(np.mean([rouge_n(pred, gold) for pred, gold in zip(preds, golds)]))
    rl = float(np.mean([rouge_l(pred, gold) for pred, gold in zip(preds, golds)]))
    return {"rouge1": r1, "rougeL": rl, "combo": 0.5 * r1 + 0.5 * rl}


def load_adapted(arch, method, rank, alpha, unfreeze, base_weights, peft_path, device):
    model = build_model(arch, vocab_size=8192)
    model.load_state_dict(torch.load(base_weights, map_location="cpu", weights_only=True))
    adapt(model, arch, method, rank, alpha, unfreeze)
    checkpoint = torch.load(peft_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("teacher load missing", len(missing), "unexpected", len(unexpected), flush=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("efficient", "decoder", "hybrid", "encdec"), default="decoder")
    parser.add_argument("--method", choices=("lorma", "lora", "adapter", "partial"), default="lorma")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--train_minutes", type=int, default=0)
    parser.add_argument("--src_tokens", type=int, default=768)
    parser.add_argument("--tgt_tokens", type=int, default=192)
    parser.add_argument("--unfreeze", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_docs", type=int, default=64)
    parser.add_argument("--gen_tokens", type=int, default=160)
    parser.add_argument("--min_gen", type=int, default=0)
    parser.add_argument("--gen_batch", type=int, default=4)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--hier", action="store_true")
    parser.add_argument("--extract_words", type=int, default=0)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--teacher_weights", default="")
    parser.add_argument("--teacher_base", default="")
    parser.add_argument("--teacher_arch", default="efficient")
    parser.add_argument("--teacher_method", default="lorma")
    parser.add_argument("--teacher_unfreeze", type=int, default=4)
    parser.add_argument("--kd_weight", type=float, default=0.5)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--decode_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.weights = args.weights or DEFAULT_WEIGHTS[args.arch]
    if args.arch == "encdec":
        args.src_tokens = min(args.src_tokens, BLOCK)
        args.tgt_tokens = min(args.tgt_tokens, BLOCK - 2)
    else:
        args.src_tokens = min(args.src_tokens, BLOCK - args.tgt_tokens - 2)

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if args.require_cuda:
        for _ in range(90):
            if torch.cuda.is_available():
                break
            time.sleep(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("GPU required")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("cuda", torch.cuda.get_device_name(0), flush=True)
    print("device", device, "arch", args.arch, "method", args.method, flush=True)

    root = os.path.join(args.data_dir, "Phase_6_Summ")
    train_rows = read_rows(os.path.join(root, "Train", "train.json"), True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), False)
    print("docs", len(train_rows), len(test_rows), flush=True)

    tokenizer = LegalBPE.load(args.bpe)
    rng = random.Random(args.seed)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    n_val = max(32, int(len(train_rows) * args.val_frac)) if not args.smoke else min(8, len(train_rows))
    n_val = min(n_val, 80)
    val_rows = [train_rows[i] for i in order[:n_val]]
    fit_rows = [train_rows[i] for i in order[n_val:]]
    if args.smoke:
        fit_rows, val_rows, test_rows = fit_rows[:32], val_rows[:4], test_rows[:8]
        args.max_steps = args.max_steps or 20
        args.gen_tokens = min(args.gen_tokens, 32)
        args.eval_docs = min(args.eval_docs, 4)
        args.batch = min(args.batch, 1)
        args.src_tokens = min(args.src_tokens, 256)
        args.tgt_tokens = min(args.tgt_tokens, 64)
    if args.max_docs:
        fit_rows = fit_rows[:args.max_docs]
    if args.decode_only or args.no_train:
        fit_rows = []
        val_rows = val_rows[: max(args.eval_docs, 8)]

    def encode_row(row):
        if "src" in row:
            return
        row["src"] = tokenizer.encode(row["document"])
        row["tgt"] = tokenizer.encode(row["summary"]) if row["summary"] else []

    todo = fit_rows + val_rows + test_rows
    print("encoding", len(todo), "docs", flush=True)
    for index, row in enumerate(todo, 1):
        encode_row(row)
        if index % 500 == 0 or index == len(todo):
            print(f"  encoded {index}/{len(todo)}", flush=True)
    for row in fit_rows:
        if args.arch == "encdec":
            row["enc"] = clip_ids(row["src"], args.src_tokens)
            row["dec"] = [BOS_ID] + row["tgt"][:args.tgt_tokens] + [EOS_ID]
        else:
            seq, labels = pack_causal(row["src"], row["tgt"], args.src_tokens, args.tgt_tokens)
            row["seq"], row["labels"] = seq, labels
    print("train", len(fit_rows), "val", len(val_rows), "src", args.src_tokens, "tgt", args.tgt_tokens, flush=True)

    model = build_model(args.arch, vocab_size=8192)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    adapt(model, args.arch, args.method, args.rank, args.alpha, args.unfreeze)
    model.to(device)
    body = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"trainable {sum(parameter.numel() for parameter in body) / 1e6:.3f}M", flush=True)
    if args.decode_only or args.no_train:
        ckpt = os.path.join(args.out, "best.pt")
        if args.decode_only and os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"], strict=False)
            model.to(device)
            print("loaded", ckpt, flush=True)
        val_used = val_rows[: max(args.eval_docs, 8)]
        val_preds = generate_rows(model, tokenizer, val_used, device, args.src_tokens, args.gen_tokens,
                                  gen_batch=args.gen_batch, hier=args.hier, min_new=args.min_gen,
                                  extract_words=args.extract_words)
        report = score_rows(val_preds, [row["summary"] for row in val_used])
        print("val report", json.dumps(report), flush=True)
        with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("generating test", len(test_rows), flush=True)
        test_preds = generate_rows(model, tokenizer, test_rows, device, args.src_tokens, args.gen_tokens,
                                   gen_batch=args.gen_batch, hier=args.hier, min_new=args.min_gen,
                                   extract_words=args.extract_words)
        predictions = [{"id": row["id"], "summary": pred} for row, pred in zip(test_rows, test_preds)]
        pred_path = os.path.join(args.out, "prediction.json")
        with open(pred_path, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle)
        zip_path = os.path.join(args.out, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(pred_path, arcname="prediction.json")
        print("zip", zip_path, "rows", len(predictions), flush=True)
        return
    resume_path = os.path.join(args.out, "best.pt")
    if args.resume and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device)
        print("resumed", resume_path, checkpoint.get("score"), flush=True)
    optimizer = torch.optim.AdamW(body, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    teacher = None
    if args.teacher_weights:
        teacher_base = args.teacher_base
        if not teacher_base:
            parent = os.path.dirname(args.weights)
            kaggle_name = os.path.join(parent, f"{args.teacher_arch}_model.pt")
            if os.path.basename(args.weights) == f"{args.arch}_model.pt" and os.path.isfile(kaggle_name):
                teacher_base = kaggle_name
            else:
                teacher_base = DEFAULT_WEIGHTS.get(args.teacher_arch, "")
        if not teacher_base or not os.path.isfile(teacher_base):
            raise SystemExit(f"missing teacher base weights for {args.teacher_arch}: {teacher_base!r}")
        teacher = load_adapted(
            args.teacher_arch, args.teacher_method, args.rank, args.alpha, args.teacher_unfreeze,
            teacher_base, args.teacher_weights, device,
        )
        print("teacher", args.teacher_arch, args.teacher_method, teacher_base, flush=True)

    steps_per_epoch = math.ceil(len(fit_rows) / max(1, args.batch))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    best_path = os.path.join(args.out, "best.pt")
    best_score = -1.0
    deadline = time.time() + args.train_minutes * 60 if args.train_minutes else 0

    def evaluate(tag):
        nonlocal best_score
        sample = val_rows[:args.eval_docs]
        preds = generate_rows(model, tokenizer, sample, device, args.src_tokens, args.gen_tokens,
                              gen_batch=args.gen_batch, hier=args.hier, min_new=args.min_gen,
                              extract_words=args.extract_words)
        golds = [row["summary"] for row in sample]
        report = score_rows(preds, golds)
        print(f"  val[{len(sample)}] {tag} rouge1 {report['rouge1']:.4f} rougeL {report['rougeL']:.4f} {preds[0][:90]!r}", flush=True)
        if report["combo"] >= best_score:
            best_score = report["combo"]
            torch.save({"args": {k: v for k, v in vars(args).items() if k != "teacher_weights" or v},
                        "score": report, "model": trainable_state(model)}, best_path)
            print("  saved", best_path, flush=True)
        model.train()
        return report

    global_step = 0
    started = time.time()
    loss_fn = encdec_loss if args.arch == "encdec" else causal_loss
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        rng.shuffle(fit_rows)
        recent = 0.0
        seen = 0
        for start in range(0, len(fit_rows), args.batch):
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
            if deadline and time.time() >= deadline:
                print("train_minutes reached", args.train_minutes, flush=True)
                stop = True
                break
            set_lrs(optimizer, global_step, total_steps, args.warmup, [args.lr])
            batch = fit_rows[start:start + args.batch]
            loss = loss_fn(model, batch, device)
            if teacher is not None:
                seqs, starts, widths, student_bits, keep_bits = [], [], [], [], []
                if args.arch == "encdec":
                    src, src_mask = pad_batch(batch, "enc", device)
                    tgt, tgt_mask = pad_batch(batch, "dec", device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        memory = model.encode(src, src_mask)
                        student_logits = model.decode(tgt[:, :-1], memory, pad_mask=tgt_mask[:, :-1], memory_mask=src_mask)
                    for index, row in enumerate(batch):
                        packed = clip_ids(row["src"], args.src_tokens) + row["dec"]
                        bos_at = len(packed) - len(row["dec"])
                        if len(packed) > BLOCK:
                            packed = packed[-BLOCK:]
                            bos_at = len(packed) - len(row["dec"])
                        seqs.append(packed)
                        starts.append(max(0, bos_at))
                        widths.append(len(row["dec"]) - 1)
                        student_bits.append(student_logits[index, :len(row["dec"]) - 1])
                        keep_bits.append(torch.ones(len(row["dec"]) - 1, dtype=torch.bool, device=device))
                else:
                    tokens, mask = pad_batch(batch, "seq", device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        student_logits, _ = model(tokens, pad_mask=mask)
                    for index, row in enumerate(batch):
                        first = next((i for i, label in enumerate(row["labels"]) if label != -100), 1)
                        bos_at = max(0, first - 1)
                        width = sum(1 for label in row["labels"] if label != -100)
                        seqs.append(row["seq"])
                        starts.append(bos_at)
                        widths.append(width)
                        student_bits.append(student_logits[index, bos_at:bos_at + width])
                        keep_bits.append(torch.ones(width, dtype=torch.bool, device=device))
                teacher_bits = teacher_causal_summary_logits(teacher, seqs, starts, widths, device)
                kd = 0.0
                count = 0
                for student_piece, teacher_piece, keep in zip(student_bits, teacher_bits, keep_bits):
                    width = min(student_piece.size(0), teacher_piece.size(0), keep.size(0))
                    if width == 0:
                        continue
                    kd = kd + kd_loss(student_piece[:width], teacher_piece[:width], keep[:width], args.kd_temp)
                    count += 1
                if count:
                    loss = (1 - args.kd_weight) * loss + args.kd_weight * (kd / count)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(body, 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            recent += loss.item()
            seen += 1
            if global_step % 50 == 0:
                print(f"  epoch {epoch} step {global_step}/{total_steps} loss {recent / max(1, seen):.4f} {time.time() - started:.0f}s", flush=True)
                recent = 0.0
                seen = 0
            if args.eval_every and global_step % args.eval_every == 0:
                evaluate(f"step {global_step}")
        evaluate(f"epoch {epoch}")
        if stop:
            break

    if os.path.isfile(best_path):
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device)
    val_used = val_rows[: max(args.eval_docs, 24)]
    val_preds = generate_rows(model, tokenizer, val_used, device, args.src_tokens, args.gen_tokens,
                              gen_batch=args.gen_batch, hier=args.hier, min_new=args.min_gen,
                              extract_words=args.extract_words)
    report = score_rows(val_preds, [row["summary"] for row in val_used])
    report["params_trainable_m"] = round(sum(parameter.numel() for parameter in body) / 1e6, 3)
    report["minutes"] = round((time.time() - started) / 60, 1)
    report["steps"] = global_step
    print("val report", json.dumps(report), flush=True)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(args.out, "val_prediction.json"), "w", encoding="utf-8") as handle:
        json.dump([{"id": row["id"], "summary": pred} for row, pred in zip(val_used, val_preds)], handle)

    print("generating test", len(test_rows), flush=True)
    test_preds = generate_rows(model, tokenizer, test_rows, device, args.src_tokens, args.gen_tokens,
                               gen_batch=args.gen_batch, hier=args.hier, min_new=args.min_gen,
                               extract_words=args.extract_words)
    predictions = [{"id": row["id"], "summary": pred} for row, pred in zip(test_rows, test_preds)]
    pred_path = os.path.join(args.out, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    zip_path = os.path.join(args.out, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    names = zipfile.ZipFile(zip_path).namelist()
    if names != ["prediction.json"]:
        raise SystemExit(f"zip layout {names}")
    print("zip", zip_path, "rows", len(predictions), flush=True)


if __name__ == "__main__":
    main()
