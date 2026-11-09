"""Phase 5 legal statute identification from the Phase 1 models.

A document is read as up to --chunks token windows. Every statute has its own
attention query over all tokens of the document (label-wise attention), plus a
mean-pooled path. Sigmoid outputs are turned into statute sets with per-statute
thresholds tuned on dev.

    python phase5_lsi.py --arch efficient --method lorma --out runs/eff_lorma
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from legal_bpe import LegalBPE
from phase1_models import build_model
from phase2_lner import enable_full_attention, hidden_states
from phase4_own import DEFAULT_BPE, DEFAULT_WEIGHTS, LOCAL, apply_method, set_lrs, set_seed

N_LABELS = 100
DEFAULT_DATA = os.path.join(LOCAL, "phase5data")


def read_rows(path, labeled):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for item in payload:
        labels = sorted({int(label) for label in item.get("labels", [])}) if labeled else []
        rows.append({"id": str(item["id"]), "text": str(item.get("text") or ""), "labels": labels})
    return rows


def windows_of(ids, count, width):
    if not ids:
        return [[1]]
    if len(ids) <= count * width:
        return [ids[start : start + width] for start in range(0, len(ids), width)]
    span = len(ids) - width
    starts = [int(round(i * span / (count - 1))) for i in range(count)] if count > 1 else [0]
    return [ids[start : start + width] for start in starts]


def prepare(rows, tokenizer, chunks, width, word_cap, cache_dir, split):
    key = hashlib.md5(f"{split}|{len(rows)}|{rows[0]['id']}|{chunks}|{width}|{word_cap}".encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"p5_{split}_{key}.pkl")
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            windows = pickle.load(handle)
    else:
        started = time.time()
        windows = []
        for row in rows:
            words = row["text"].split()[:word_cap]
            windows.append(windows_of(tokenizer.encode(" ".join(words)), chunks, width))
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(windows, handle)
        print(f"tokenized {split} {len(rows)} docs {time.time() - started:.0f}s", flush=True)
    for row, win in zip(rows, windows):
        row["windows"] = win
    return rows


class StatuteModel(nn.Module):
    def __init__(self, backbone, n_labels, prior_logits):
        super().__init__()
        self.backbone = backbone
        width = backbone.embed.weight.size(1)
        self.width = width
        self.query = nn.Parameter(torch.randn(n_labels, width) * 0.02)
        self.label_out = nn.Parameter(torch.randn(n_labels, width) * 0.02)
        self.mean_head = nn.Linear(width, n_labels)
        nn.init.zeros_(self.mean_head.weight)
        with torch.no_grad():
            self.mean_head.bias.copy_(prior_logits)
        self.hier = False
        self.win_query = nn.Parameter(torch.randn(n_labels, width) * 0.02)
        self.win_out = nn.Parameter(torch.randn(n_labels, width) * 0.02)

    def encode_windows(self, windows, device):
        order = sorted(range(len(windows)), key=lambda index: len(windows[index]))
        out = [None] * len(windows)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and (stop - start + 1) * len(windows[order[stop]]) <= 4096:
                stop += 1
            chunk = order[start:stop]
            longest = max(len(windows[index]) for index in chunk)
            token_ids = torch.zeros(len(chunk), longest, dtype=torch.long, device=device)
            pad_mask = torch.zeros(len(chunk), longest, dtype=torch.bool, device=device)
            for row, index in enumerate(chunk):
                ids = windows[index]
                token_ids[row, : len(ids)] = torch.tensor(ids, device=device)
                pad_mask[row, : len(ids)] = True
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = hidden_states(self.backbone, token_ids, pad_mask)
            for row, index in enumerate(chunk):
                out[index] = hidden[row, : len(windows[index])]
            start = stop
        return out

    def forward(self, batch_windows, device):
        flat = [window for windows in batch_windows for window in windows]
        pieces = self.encode_windows(flat, device)
        docs, win_means = [], []
        start = 0
        for windows in batch_windows:
            chunk = pieces[start : start + len(windows)]
            docs.append(torch.cat(chunk, dim=0))
            win_means.append(torch.stack([item.float().mean(0) for item in chunk]))
            start += len(windows)
        longest = max(doc.size(0) for doc in docs)
        hidden = torch.zeros(len(docs), longest, self.width, device=device)
        mask = torch.zeros(len(docs), longest, dtype=torch.bool, device=device)
        for row, doc in enumerate(docs):
            hidden[row, : doc.size(0)] = doc.float()
            mask[row, : doc.size(0)] = True
        scores = torch.einsum("btd,ld->btl", hidden, self.query) / math.sqrt(self.width)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
        weight = torch.softmax(scores, dim=1)
        per_label = torch.einsum("btl,btd->bld", weight, hidden)
        attended = (per_label * self.label_out).sum(-1)
        mean = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        logits = attended + self.mean_head(mean)
        if self.hier:
            widest = max(item.size(0) for item in win_means)
            windows = torch.zeros(len(win_means), widest, self.width, device=device)
            win_mask = torch.zeros(len(win_means), widest, dtype=torch.bool, device=device)
            for row, item in enumerate(win_means):
                windows[row, : item.size(0)] = item
                win_mask[row, : item.size(0)] = True
            win_scores = torch.einsum("bwd,ld->bwl", windows, self.win_query) / math.sqrt(self.width)
            win_scores = win_scores.masked_fill(~win_mask.unsqueeze(-1), -1e4)
            win_weight = torch.softmax(win_scores, dim=1)
            win_vec = torch.einsum("bwl,bwd->bld", win_weight, windows)
            logits = logits + (win_vec * self.win_out).sum(-1)
        return logits


def build(arch, method, weights, rank, alpha, unfreeze, prior):
    backbone = build_model(arch, vocab_size=8192)
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    enable_full_attention(backbone)
    apply_method(backbone, arch, method, rank, alpha, unfreeze)
    return StatuteModel(backbone, N_LABELS, prior)


def trainable_state(model):
    named = dict(model.named_parameters())
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.") or (name in named and named[name].requires_grad)
    }


def label_matrix(rows):
    target = np.zeros((len(rows), N_LABELS), dtype=np.float32)
    for index, row in enumerate(rows):
        target[index, row["labels"]] = 1.0
    return target


def base_of(batch, device):
    if "base" not in batch[0]:
        return 0.0
    return torch.tensor(np.stack([row["base"] for row in batch]), dtype=torch.float32, device=device)


def attach_base(folder, splits):
    """Fixed logits (e.g. out-of-fold TF-IDF) that the model learns to correct."""
    for split, rows in splits:
        with open(os.path.join(folder, f"{split}_logits_ids.json"), encoding="utf-8") as handle:
            ids = json.load(handle)
        values = np.load(os.path.join(folder, f"{split}_logits.npy")).astype(np.float32)
        where = {item: index for index, item in enumerate(ids)}
        for row in rows:
            row["base"] = values[where[row["id"]]]


def logits_of(model, rows, device, batch_size=8):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            logits = model([row["windows"] for row in batch], device).float() + base_of(batch, device)
            out.append(logits.cpu().numpy())
    return np.concatenate(out, axis=0)


def macro_f1(gold, pred):
    tp = (gold * pred).sum(0)
    fp = ((1 - gold) * pred).sum(0)
    fn = (gold * (1 - pred)).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def micro_f1(gold, pred):
    tp = (gold * pred).sum()
    return float(2 * tp / max(1.0, gold.sum() + pred.sum()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def asymmetric_loss(logits, target, gamma_neg, clip):
    """Asymmetric loss for multi-label data: hard negatives are down-weighted, easy negatives are clipped."""
    prob = torch.sigmoid(logits)
    neg = (1 - prob + clip).clamp(max=1)
    loss = target * torch.log(prob.clamp(min=1e-6)) + (1 - target) * torch.log(neg.clamp(min=1e-6))
    focus = prob * target + neg * (1 - target)
    weight = (1 - focus).pow(gamma_neg * (1 - target))
    return -(weight * loss).mean()


GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)


def per_label_cuts(gold, probs):
    cuts = np.zeros(gold.shape[1])
    for label in range(gold.shape[1]):
        pred = probs[:, label][:, None] >= GRID[None, :]
        y = gold[:, label][:, None]
        tp = (pred * y).sum(0)
        fp = (pred * (1 - y)).sum(0)
        fn = ((~pred) * y).sum(0)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        cuts[label] = GRID[int(np.argmax(f1))]
    return cuts


def global_cut(gold, probs):
    scores = [macro_f1(gold, (probs >= cut).astype(np.float32)) for cut in GRID]
    best = int(np.argmax(scores))
    return float(GRID[best]), float(scores[best])


def apply_cuts(probs, cuts, at_least_one=True):
    pred = (probs >= cuts[None, :]).astype(np.float32)
    if at_least_one:
        empty = pred.sum(1) == 0
        ratio = probs / np.maximum(cuts[None, :], 1e-6)
        pred[empty, ratio[empty].argmax(1)] = 1.0
    return pred


def threshold_report(gold, probs, seed=0):
    """Per-label cuts judged honestly: tune on one half of dev, score the other half."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(gold))
    halves = [order[: len(order) // 2], order[len(order) // 2 :]]
    honest = []
    for tune, hold in (halves, halves[::-1]):
        cuts = per_label_cuts(gold[tune], probs[tune])
        honest.append(macro_f1(gold[hold], apply_cuts(probs[hold], cuts)))
    g_cut, g_f1 = global_cut(gold, probs)
    return {
        "f1_at_0.5": macro_f1(gold, (probs >= 0.5).astype(np.float32)),
        "global_cut": g_cut,
        "f1_global_cut": g_f1,
        "f1_per_label_crossval": float(np.mean(honest)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("efficient", "decoder", "hybrid", "encdec"), default="efficient")
    parser.add_argument("--method", choices=("lorma", "lora", "adapter", "bitfit", "partial"), default="lorma")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--cache_dir", default=os.path.join(LOCAL, "phase5_cache"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--chunk_tokens", type=int, default=256)
    parser.add_argument("--word_cap", type=int, default=2500)
    parser.add_argument("--unfreeze", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--pos_weight", type=float, default=0.0, help="power on neg/pos ratio for BCE pos_weight")
    parser.add_argument("--loss", choices=("bce", "asl"), default="bce")
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--eval_docs", type=int, default=2500)
    parser.add_argument("--teacher_logits", default="")
    parser.add_argument("--kd_weight", type=float, default=0.5)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--dump_train", action="store_true")
    parser.add_argument("--base_logits", default="", help="folder with {train,dev,test}_logits.npy to add to the output")
    parser.add_argument("--hier", action="store_true", help="also attend over window means, HierBERT-style")
    parser.add_argument("--include_dev", action="store_true", help="train on labeled train+dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.weights = args.weights or DEFAULT_WEIGHTS[args.arch]

    set_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print("device", device, "arch", args.arch, "method", args.method, "kd", bool(args.teacher_logits), flush=True)

    root = os.path.join(args.data_dir, "Phase_5_LSI")
    train_rows = read_rows(os.path.join(root, "Train", "train.json"), True)
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), False)
    print("docs", len(train_rows), len(dev_rows), len(test_rows), flush=True)

    tokenizer = LegalBPE.load(args.bpe)
    prepare(train_rows, tokenizer, args.chunks, args.chunk_tokens, args.word_cap, args.cache_dir, "train")
    prepare(dev_rows, tokenizer, args.chunks, args.chunk_tokens, args.word_cap, args.cache_dir, "dev")
    prepare(test_rows, tokenizer, args.chunks, args.chunk_tokens, args.word_cap, args.cache_dir, "test")

    if args.base_logits:
        attach_base(args.base_logits, (("train", train_rows), ("dev", dev_rows), ("test", test_rows)))
        print("base logits from", args.base_logits, flush=True)
    if args.include_dev:
        train_rows = list(train_rows) + [dict(row) for row in dev_rows]
        print("training on train+dev", len(train_rows), flush=True)
    if args.smoke:
        dev_rows, test_rows = dev_rows[:200], test_rows[:50]

    teacher = None
    if args.teacher_logits:
        with open(args.teacher_logits.replace(".npy", "_ids.json"), encoding="utf-8") as handle:
            teacher_ids = json.load(handle)
        values = np.load(args.teacher_logits)
        teacher = {item: values[index] for index, item in enumerate(teacher_ids)}
        print("teacher logits", values.shape, flush=True)

    full_train = list(train_rows)
    random.shuffle(train_rows)
    if args.max_docs:
        train_rows = train_rows[: args.max_docs]
    target = label_matrix(train_rows)
    freq = target.mean(0).clip(1e-4, 1 - 1e-4)
    prior = torch.tensor(np.log(freq / (1 - freq)), dtype=torch.float32)
    if args.base_logits:
        base = np.stack([row["base"] for row in train_rows])
        low, high = np.full(N_LABELS, -20.0), np.full(N_LABELS, 20.0)
        for _ in range(40):
            mid = (low + high) / 2
            too_high = sigmoid(base + mid[None, :]).mean(0) > freq
            high = np.where(too_high, mid, high)
            low = np.where(too_high, low, mid)
        prior = torch.tensor((low + high) / 2, dtype=torch.float32)
    pos_weight = None
    if args.pos_weight > 0:
        pos_weight = torch.tensor(((1 - freq) / freq) ** args.pos_weight, dtype=torch.float32, device=device)
    print("train subset", len(train_rows), "labels per doc", round(float(target.sum(1).mean()), 2), flush=True)

    model = build(args.arch, args.method, args.weights, args.rank, args.alpha, args.unfreeze, prior).to(device)
    model.hier = args.hier
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    print(f"trainable head {sum(p.numel() for p in head) / 1e6:.3f}M body {sum(p.numel() for p in body) / 1e6:.3f}M", flush=True)
    groups = [{"params": head, "lr": args.head_lr, "weight_decay": 0.0}]
    if body:
        groups.append({"params": body, "lr": args.lr, "weight_decay": 0.01})
    optimizer = torch.optim.AdamW(groups)
    base_lrs = [group["lr"] for group in groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    steps_per_epoch = math.ceil(len(train_rows) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    eval_rows = dev_rows[: args.eval_docs]
    eval_gold = label_matrix(eval_rows)
    best_path = os.path.join(args.out, "best.pt")
    best_score = -1.0

    def evaluate(tag):
        nonlocal best_score
        probs = sigmoid(logits_of(model, eval_rows, device))
        cut, score = global_cut(eval_gold, probs)
        print(f"  dev[{len(eval_rows)}] {tag} macroF1@0.5 {macro_f1(eval_gold, (probs >= 0.5).astype(np.float32)):.4f} best global {score:.4f} at {cut}", flush=True)
        if score > best_score:
            best_score = score
            torch.save({"args": vars(args), "score": score, "model": trainable_state(model)}, best_path)
            print("  saved", best_path, flush=True)
        model.train()

    global_step = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for module in model.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        recent = 0.0
        for start in range(0, len(order), args.batch):
            if args.max_steps and global_step >= args.max_steps:
                break
            set_lrs(optimizer, global_step, total_steps, args.warmup, base_lrs)
            chosen = order[start : start + args.batch]
            batch = [train_rows[index] for index in chosen]
            logits = model([row["windows"] for row in batch], device) + base_of(batch, device)
            y = torch.tensor(target[chosen], device=device)
            if args.loss == "asl":
                loss = asymmetric_loss(logits, y, args.asl_gamma_neg, args.asl_clip)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            if teacher is not None:
                soft = torch.tensor(np.stack([teacher[row["id"]] for row in batch]), dtype=torch.float32, device=device)
                temp = args.kd_temp
                kd = F.binary_cross_entropy_with_logits(logits / temp, torch.sigmoid(soft / temp)) * temp * temp
                loss = (1 - args.kd_weight) * loss + args.kd_weight * kd
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head + body, 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            recent += loss.item()
            if global_step % 100 == 0:
                print(f"  epoch {epoch} step {global_step}/{total_steps} loss {recent / 100:.4f} {time.time() - started:.0f}s", flush=True)
                recent = 0.0
            if args.eval_every and global_step % args.eval_every == 0:
                evaluate(f"step {global_step}")
        evaluate(f"epoch {epoch}")

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    dev_gold = label_matrix(dev_rows)
    dev_logits = logits_of(model, dev_rows, device)
    test_logits = logits_of(model, test_rows, device)
    dev_probs, test_probs = sigmoid(dev_logits), sigmoid(test_logits)
    report = threshold_report(dev_gold, dev_probs)
    cuts = per_label_cuts(dev_gold, dev_probs)
    final_dev = apply_cuts(dev_probs, cuts)
    report["f1_per_label_on_dev_optimistic"] = macro_f1(dev_gold, final_dev)
    report["micro_f1_per_label"] = micro_f1(dev_gold, final_dev)
    report["params_trainable_m"] = round(sum(p.numel() for p in head + body) / 1e6, 3)
    report["minutes"] = round((time.time() - started) / 60, 1)
    print("dev report", json.dumps(report), flush=True)

    np.save(os.path.join(args.out, "dev_logits.npy"), dev_logits.astype(np.float16))
    np.save(os.path.join(args.out, "test_logits.npy"), test_logits.astype(np.float16))
    with open(os.path.join(args.out, "dev_logits_ids.json"), "w", encoding="utf-8") as handle:
        json.dump([row["id"] for row in dev_rows], handle)
    with open(os.path.join(args.out, "test_logits_ids.json"), "w", encoding="utf-8") as handle:
        json.dump([row["id"] for row in test_rows], handle)
    if args.dump_train:
        train_logits = logits_of(model, full_train, device)
        np.save(os.path.join(args.out, "train_logits.npy"), train_logits.astype(np.float32))
        with open(os.path.join(args.out, "train_logits_ids.json"), "w", encoding="utf-8") as handle:
            json.dump([row["id"] for row in full_train], handle)
        print("dumped train logits", train_logits.shape, flush=True)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(args.out, "thresholds.json"), "w", encoding="utf-8") as handle:
        json.dump([float(cut) for cut in cuts], handle)

    test_pred = apply_cuts(test_probs, cuts)
    predictions = [
        {"id": row["id"], "pred": [int(label) for label in np.flatnonzero(test_pred[index])]}
        for index, row in enumerate(test_rows)
    ]
    pred_path = os.path.join(args.out, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    with zipfile.ZipFile(os.path.join(args.out, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("zip", os.path.join(args.out, "submission.zip"), "rows", len(predictions), flush=True)


if __name__ == "__main__":
    main()
