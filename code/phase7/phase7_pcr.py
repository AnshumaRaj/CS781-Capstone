"""Phase 7 prior-case retrieval from the four Phase 1 models (+ TF-IDF).

No public BERT. Lexical retrieval uses only this dataset. Neural retrieval
mean-pools Phase 1 hidden states over a few windows.

    python phase7_pcr.py --mode tfidf --out ...
    python phase7_pcr.py --mode neural --arch efficient --method lorma --require_cuda
    python phase7_pcr.py --mode blend --arch efficient --method lorma --require_cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
CAP = HERE.parents[1]
for extra in (CAP / "phase1" / "code", CAP / "phase2" / "code", CAP / "phase4" / "code"):
    sys.path.insert(0, str(extra))

import torch
import torch.nn.functional as F

from legal_bpe import PAD_ID, LegalBPE
from phase1_models import build_model
from phase2_lner import hidden_states
from phase4_own import DEFAULT_BPE, DEFAULT_WEIGHTS, LOCAL, apply_method, set_seed, windows_of

DEFAULT_DATA = os.path.join(LOCAL, "phase7data")
K = 7
POOL_K = 50


def join_text(value):
    if isinstance(value, dict):
        value = value.get("text") or value.get("document") or ""
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return str(value or "")


def clip_words(text, cap):
    words = text.split()
    if len(words) <= cap:
        return text
    head = cap * 2 // 3
    return " ".join(words[:head] + words[-(cap - head):])


def load_queries(path, labeled):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for key, value in payload.items():
        text = join_text(value)
        rel = []
        if labeled and isinstance(value, dict):
            rel = [str(item) for item in value.get("relevant_candidates") or [] if str(item).strip()]
        rows.append({"id": str(key), "text": text, "rel": rel})
    return rows


def load_candidates(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): join_text(value) for key, value in payload.items()}


def micro_at_k(queries, ranked, k=K):
    tp = fp = fn = 0
    for row in queries:
        gold = set(row["rel"])
        pred = ranked[row["id"]][:k]
        got = set(pred)
        tp += len(gold & got)
        fp += len(got - gold)
        fn += len(gold - got)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def topk(scores, cand_ids, query_id, k=POOL_K):
    order = np.argsort(-np.asarray(scores))
    out = []
    for index in order:
        cid = cand_ids[index]
        if cid == query_id:
            continue
        out.append(cid)
        if len(out) >= k:
            break
    return out


def cut(ranked, k=K):
    return {qid: ids[:k] for qid, ids in ranked.items()}


def tfidf_rank(queries, candidates, word_cap, ngram, features):
    ids = list(candidates)
    docs = [clip_words(candidates[cid], word_cap) for cid in ids]
    qtexts = [clip_words(row["text"], word_cap) for row in queries]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram),
        max_features=features,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    matrix = normalize(vectorizer.fit_transform(docs + qtexts))
    cand_x = matrix[:len(ids)]
    query_x = matrix[len(ids):]
    ranked = {}
    dense_scores = (query_x @ cand_x.T).toarray()
    for row, scores in zip(queries, dense_scores):
        ranked[row["id"]] = topk(scores, ids, row["id"])
    return ranked, dense_scores, ids


def write_zip(path, ranked):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pred_path = os.path.join(os.path.dirname(path), "prediction.json")
    payload = cut(ranked)
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("zip", path, "queries", len(payload), flush=True)


def encode_many(model, tokenizer, texts, device, windows, width, train=False):
    all_chunks = []
    owners = []
    for index, text in enumerate(texts):
        ids = tokenizer.encode(clip_words(text, 4000)) or [1]
        for chunk in windows_of(ids, windows, width):
            all_chunks.append(chunk or [1])
            owners.append(index)
    max_len = max(len(chunk) for chunk in all_chunks)
    tensor = torch.tensor(
        [chunk + [PAD_ID] * (max_len - len(chunk)) for chunk in all_chunks],
        dtype=torch.long,
        device=device,
    )
    mask = tensor != PAD_ID
    context = nullcontext() if train else torch.no_grad()
    with context:
        hidden = hidden_states(model, tensor, mask)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        pooled = F.normalize(pooled.float(), dim=-1)
        vecs = []
        owners_t = torch.tensor(owners, device=device)
        for index in range(len(texts)):
            parts = pooled[owners_t == index]
            vecs.append(F.normalize(parts.mean(0), dim=-1))
        return torch.stack(vecs, 0)


def encode_pool(model, tokenizer, candidates, device, windows, width, pool_batch):
    ids = list(candidates)
    texts = [candidates[cid] for cid in ids]
    chunks = []
    for start in range(0, len(texts), pool_batch):
        chunk = encode_many(
            model, tokenizer, texts[start:start + pool_batch], device, windows, width, train=False
        ).cpu()
        chunks.append(chunk)
        done = min(start + pool_batch, len(texts))
        if done % 200 == 0 or done == len(texts):
            print(f"  encoded {done}/{len(texts)}", flush=True)
    return ids, F.normalize(torch.cat(chunks, 0), dim=-1)


def neural_rank(model, tokenizer, queries, candidates, device, windows, width, pool_batch):
    cand_ids, cand_vecs = encode_pool(model, tokenizer, candidates, device, windows, width, pool_batch)
    ranked = {}
    q_texts = [row["text"] for row in queries]
    q_chunks = []
    for start in range(0, len(q_texts), pool_batch):
        q_chunks.append(
            encode_many(
                model, tokenizer, q_texts[start:start + pool_batch], device, windows, width, train=False
            ).cpu()
        )
    query_vecs = F.normalize(torch.cat(q_chunks, 0), dim=-1)
    scores = (query_vecs @ cand_vecs.T).numpy()
    for row, row_scores in zip(queries, scores):
        ranked[row["id"]] = topk(row_scores, cand_ids, row["id"])
    return ranked, scores, cand_ids


def blend_rank(lex, neural, queries, lex_w):
    out = {}
    for row in queries:
        qid = row["id"]
        scores = {}
        for rank, cid in enumerate(lex[qid]):
            scores[cid] = scores.get(cid, 0.0) + lex_w / (rank + 1)
        for rank, cid in enumerate(neural[qid]):
            scores[cid] = scores.get(cid, 0.0) + (1.0 - lex_w) / (rank + 1)
        ordered = sorted(scores, key=scores.get, reverse=True)
        out[qid] = [cid for cid in ordered if cid != qid][:POOL_K]
    return out


def sample_negs(row, candidates, negs):
    banned = set(row["rel"]) | {row["id"]}
    hard = [cid for cid in row.get("_hard") or [] if cid not in banned]
    out = []
    if len(hard) >= negs:
        return random.sample(hard, negs)
    out.extend(hard)
    keys = list(candidates)
    while len(out) < negs:
        cid = random.choice(keys)
        if cid not in banned and cid not in out:
            out.append(cid)
    return out


def row_z(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def rerank_head(lex_ranked, lex_scores, lex_ids, neu_scores, neu_ids, queries, topn, mix):
    lex_index = {cid: index for index, cid in enumerate(lex_ids)}
    neu_index = {cid: index for index, cid in enumerate(neu_ids)}
    out = {}
    for qindex, row in enumerate(queries):
        head = lex_ranked[row["id"]][:topn]
        rest = lex_ranked[row["id"]][topn:]
        lex_part = np.array([lex_scores[qindex, lex_index[cid]] for cid in head], dtype=np.float32)
        neu_part = np.array([neu_scores[qindex, neu_index[cid]] for cid in head], dtype=np.float32)
        fused = row_z(lex_part) + mix * row_z(neu_part)
        order = np.argsort(-fused)
        out[row["id"]] = [head[index] for index in order] + rest
    return out


def scores_from(query, positive, negative):
    pos_score = (query * positive).sum(-1, keepdim=True)
    neg_score = torch.einsum("bd,bnd->bn", query, negative)
    return torch.cat([pos_score, neg_score], dim=1)


def contrastive_step(model, tokenizer, batch, candidates, device, windows, width, negs, teacher=None, kd_weight=0.0, kd_temp=2.0):
    pos_ids = []
    neg_ids = []
    for row in batch:
        if row["rel"]:
            pos_ids.append(random.choice(row["rel"]))
        else:
            pos_ids.append(random.choice(list(candidates)))
        neg_ids.extend(sample_negs(row, candidates, negs))
    q_texts = [row["text"] for row in batch]
    p_texts = [candidates[cid] for cid in pos_ids]
    n_texts = [candidates[cid] for cid in neg_ids]
    query = encode_many(model, tokenizer, q_texts, device, windows, width, train=True)
    positive = encode_many(model, tokenizer, p_texts, device, windows, width, train=True)
    negative = encode_many(model, tokenizer, n_texts, device, windows, width, train=True).view(len(batch), negs, -1)
    logits = scores_from(query, positive, negative) / 0.07
    target = torch.zeros(len(batch), dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, target)
    if teacher is None or kd_weight <= 0:
        return loss
    with torch.no_grad():
        t_query = encode_many(teacher, tokenizer, q_texts, device, windows, width, train=False)
        t_pos = encode_many(teacher, tokenizer, p_texts, device, windows, width, train=False)
        t_neg = encode_many(teacher, tokenizer, n_texts, device, windows, width, train=False).view(len(batch), negs, -1)
        t_logits = scores_from(t_query, t_pos, t_neg) / kd_temp
    kd = F.kl_div(F.log_softmax(logits / kd_temp, dim=-1), F.softmax(t_logits, dim=-1), reduction="batchmean")
    return (1.0 - kd_weight) * loss + kd_weight * kd


class PairHead(torch.nn.Module):
    def __init__(self, width, extra=2):
        super().__init__()
        self.drop = torch.nn.Dropout(0.1)
        self.proj = torch.nn.Linear(width + extra, 1)

    def forward(self, pooled, extra):
        return self.proj(self.drop(torch.cat([pooled, extra], dim=-1))).squeeze(-1)


def head_words(text, cap=600):
    return " ".join(text.split()[:cap])


def pair_encode(model, tokenizer, qtexts, dtexts, device, qlen, dlen):
    q_toks, d_toks = [], []
    for query, doc in zip(qtexts, dtexts):
        q_toks.append((tokenizer.encode(head_words(query)) or [1])[:qlen])
        d_toks.append((tokenizer.encode(head_words(doc)) or [1])[:dlen])
    max_q = max(len(ids) for ids in q_toks)
    max_d = max(len(ids) for ids in d_toks)
    rows = []
    for q_ids, d_ids in zip(q_toks, d_toks):
        rows.append(q_ids + [PAD_ID] * (max_q - len(q_ids)) + d_ids + [PAD_ID] * (max_d - len(d_ids)))
    tensor = torch.tensor(rows, dtype=torch.long, device=device)
    mask = tensor != PAD_ID
    hidden = hidden_states(model, tensor, mask)
    doc_mask = mask.clone()
    doc_mask[:, :max_q] = False
    pooled = (hidden * doc_mask.unsqueeze(-1)).sum(1) / doc_mask.sum(1).clamp(min=1).unsqueeze(-1)
    return pooled


def pair_examples(queries, candidates, scores, ids, topn, n_neg):
    index = {cid: i for i, cid in enumerate(ids)}
    items = []
    for qindex, row in enumerate(queries):
        gold = [cid for cid in row["rel"] if cid in index and cid != row["id"]]
        ranked = topk(scores[qindex], ids, row["id"], k=topn)
        rank_of = {cid: rank for rank, cid in enumerate(ranked)}
        banned = set(row["rel"]) | {row["id"]}
        negs = [cid for cid in ranked if cid not in banned][:n_neg]
        for cid in gold:
            rank = rank_of.get(cid, topn)
            items.append((row["text"], candidates[cid], 1.0, float(scores[qindex, index[cid]]), 1.0 / (rank + 1)))
        for cid in negs:
            rank = rank_of[cid]
            items.append((row["text"], candidates[cid], 0.0, float(scores[qindex, index[cid]]), 1.0 / (rank + 1)))
    return items


def pair_rerank(model, head, tokenizer, queries, candidates, lex_ranked, lex_scores, lex_ids, device, qlen, dlen, topn, batch):
    index = {cid: i for i, cid in enumerate(lex_ids)}
    out = {}
    model.eval()
    head.eval()
    for qindex, row in enumerate(queries):
        head_ids = lex_ranked[row["id"]][:topn]
        rest = lex_ranked[row["id"]][topn:]
        logits = []
        with torch.no_grad():
            for start in range(0, len(head_ids), batch):
                chunk = head_ids[start:start + batch]
                pooled = pair_encode(
                    model, tokenizer, [row["text"]] * len(chunk),
                    [candidates[cid] for cid in chunk], device, qlen, dlen,
                )
                extra = torch.tensor(
                    [[float(lex_scores[qindex, index[cid]]), 1.0 / (rank + 1)]
                     for rank, cid in enumerate(chunk, start)],
                    dtype=pooled.dtype, device=device,
                )
                logits.extend(head(pooled, extra).float().cpu().tolist())
        order = np.argsort(-np.asarray(logits, dtype=np.float32))
        out[row["id"]] = [head_ids[i] for i in order] + rest
        if (qindex + 1) % 40 == 0 or qindex + 1 == len(queries):
            print(f"  pair scored {qindex + 1}/{len(queries)}", flush=True)
    return out


def train_pair(model, head, tokenizer, items, device, qlen, dlen, batch, lr, epochs, train_minutes):
    body = [parameter for parameter in list(model.parameters()) + list(head.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(body, lr=lr, weight_decay=0.01)
    pos = sum(item[2] for item in items)
    neg = max(len(items) - pos, 1)
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
    print("pair examples", len(items), "pos", int(pos), "pos_weight", round(float(pos_weight), 2), flush=True)
    deadline = time.time() + train_minutes * 60 if train_minutes else 0
    started = time.time()
    step = 0
    stop = False
    for epoch in range(1, epochs + 1):
        random.shuffle(items)
        model.train()
        head.train()
        for start in range(0, len(items), batch):
            if deadline and time.time() >= deadline:
                stop = True
                break
            chunk = items[start:start + batch]
            pooled = pair_encode(
                model, tokenizer, [row[0] for row in chunk], [row[1] for row in chunk], device, qlen, dlen,
            )
            extra = torch.tensor([[row[3], row[4]] for row in chunk], dtype=pooled.dtype, device=device)
            logits = head(pooled, extra)
            target = torch.tensor([row[2] for row in chunk], dtype=logits.dtype, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(body, 1.0)
            optimizer.step()
            step += 1
            if step % 40 == 0:
                print(f"  pair epoch {epoch} step {step} loss {loss.item():.4f} {time.time()-started:.0f}s", flush=True)
        if stop:
            break
    model.eval()
    head.eval()


def load_backbone(arch, weights, method, rank, alpha, unfreeze, device):
    model = build_model(arch, vocab_size=8192)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    apply_method(model, arch, method, rank, alpha, unfreeze)
    return model.to(device)


def maybe_load_peft(model, path):
    if not path or not os.path.isfile(path):
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state" in state:
        state = state["state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("loaded", path, "missing", len(missing), "unexpected", len(unexpected), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tfidf", "neural", "blend", "pair"), default="tfidf")
    parser.add_argument("--arch", choices=("efficient", "decoder", "hybrid", "encdec"), default="efficient")
    parser.add_argument("--method", choices=("lorma", "lora", "adapter", "partial"), default="lorma")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--unfreeze", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--negs", type=int, default=4)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--pool_batch", type=int, default=8)
    parser.add_argument("--word_cap", type=int, default=3000)
    parser.add_argument("--ngram", type=int, default=2)
    parser.add_argument("--features", type=int, default=200000)
    parser.add_argument("--lex_w", type=float, default=0.65)
    parser.add_argument("--train_minutes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--teacher_weights", default="")
    parser.add_argument("--teacher_base", default="")
    parser.add_argument("--teacher_arch", default="efficient")
    parser.add_argument("--teacher_method", default="lorma")
    parser.add_argument("--kd_weight", type=float, default=0.0)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--hard_negs", action="store_true")
    parser.add_argument("--rerank_top", type=int, default=0)
    parser.add_argument("--rerank_mix", type=float, default=0.35)
    parser.add_argument("--qlen", type=int, default=384)
    parser.add_argument("--dlen", type=int, default=384)
    args = parser.parse_args()
    args.weights = args.weights or DEFAULT_WEIGHTS[args.arch]
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    root = os.path.join(args.data_dir, "Phase_7_PCR")
    train_q = load_queries(os.path.join(root, "Train", "train.json"), True)
    train_c = load_candidates(os.path.join(root, "Train", "train_candidates.json"))
    dev_q = load_queries(os.path.join(root, "Dev", "dev.json"), True)
    dev_c = load_candidates(os.path.join(root, "Dev", "dev_candidates.json"))
    test_q = load_queries(os.path.join(root, "Test", "test.json"), False)
    test_c = load_candidates(os.path.join(root, "Test", "test_candidates.json"))
    print("train", len(train_q), len(train_c), "dev", len(dev_q), len(dev_c), "test", len(test_q), len(test_c), flush=True)

    with open(os.path.join(args.out, "candidate_ids.json"), "w", encoding="utf-8") as handle:
        json.dump(list(test_c), handle)
    with open(os.path.join(args.out, "dev_candidate_ids.json"), "w", encoding="utf-8") as handle:
        json.dump(list(dev_c), handle)

    lex_dev, lex_dev_s, lex_dev_ids = tfidf_rank(dev_q, dev_c, args.word_cap, args.ngram, args.features)
    lex_test, lex_test_s, lex_test_ids = tfidf_rank(test_q, test_c, args.word_cap, args.ngram, args.features)
    lex_report = micro_at_k(dev_q, lex_dev)
    print("tfidf dev", json.dumps(lex_report), flush=True)
    if args.hard_negs:
        print("tfidf train hard negatives", flush=True)
        _, train_s, train_ids = tfidf_rank(train_q, train_c, args.word_cap, args.ngram, args.features)
        for row, scores in zip(train_q, train_s):
            row["_hard"] = topk(scores, train_ids, row["id"], k=80)

    ranked_dev, ranked_test = lex_dev, lex_test
    if args.mode == "pair":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.require_cuda and device.type != "cuda":
            raise SystemExit("GPU required")
        print("device", device, "pair rerank", flush=True)
        tokenizer = LegalBPE.load(args.bpe)
        model = load_backbone(args.arch, args.weights, args.method, args.rank, args.alpha, args.unfreeze, device)
        maybe_load_peft(model, args.resume)
        head = PairHead(model.embed.weight.size(1)).to(device)
        print("trainable", round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3), "M + head", flush=True)
        print("tfidf train pairs", flush=True)
        lex_train, lex_train_s, lex_train_ids = tfidf_rank(train_q, train_c, args.word_cap, args.ngram, args.features)
        topn = args.rerank_top or 30
        items = pair_examples(train_q, train_c, lex_train_s, lex_train_ids, max(topn, 50), args.negs)
        train_pair(model, head, tokenizer, items, device, args.qlen, args.dlen, args.batch, args.lr, args.epochs, args.train_minutes)
        print("pair scoring dev", flush=True)
        ranked_dev = pair_rerank(
            model, head, tokenizer, dev_q, dev_c, lex_dev, lex_dev_s, lex_dev_ids,
            device, args.qlen, args.dlen, topn, args.batch,
        )
        print("pair scoring test", flush=True)
        ranked_test = pair_rerank(
            model, head, tokenizer, test_q, test_c, lex_test, lex_test_s, lex_test_ids,
            device, args.qlen, args.dlen, topn, args.batch,
        )
        torch.save({"backbone": model.state_dict(), "head": head.state_dict()}, os.path.join(args.out, "best.pt"))
    elif args.mode != "tfidf":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.require_cuda and device.type != "cuda":
            raise SystemExit("GPU required")
        print("device", device, flush=True)
        tokenizer = LegalBPE.load(args.bpe)
        model = load_backbone(args.arch, args.weights, args.method, args.rank, args.alpha, args.unfreeze, device)
        maybe_load_peft(model, args.resume)
        body = [parameter for parameter in model.parameters() if parameter.requires_grad]
        print("trainable", round(sum(parameter.numel() for parameter in body) / 1e6, 3), "M", flush=True)
        teacher = None
        if args.kd_weight > 0 and args.teacher_weights:
            teacher_base = args.teacher_base or DEFAULT_WEIGHTS.get(args.teacher_arch, args.weights)
            teacher = load_backbone(
                args.teacher_arch, teacher_base, args.teacher_method, args.rank, args.alpha, args.unfreeze, device,
            )
            maybe_load_peft(teacher, args.teacher_weights)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            print("teacher", args.teacher_arch, args.teacher_method, flush=True)
        optimizer = torch.optim.AdamW(body, lr=args.lr, weight_decay=0.01)
        deadline = time.time() + args.train_minutes * 60 if args.train_minutes else 0
        fit = [row for row in train_q if row["rel"]]
        model.train()
        step = 0
        started = time.time()
        stop = False
        for epoch in range(1, args.epochs + 1):
            random.shuffle(fit)
            for start in range(0, len(fit), args.batch):
                if deadline and time.time() >= deadline:
                    stop = True
                    break
                batch = fit[start:start + args.batch]
                loss = contrastive_step(
                    model, tokenizer, batch, train_c, device, args.windows, args.width, args.negs,
                    teacher=teacher, kd_weight=args.kd_weight, kd_temp=args.kd_temp,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(body, 1.0)
                optimizer.step()
                step += 1
                if step % 20 == 0:
                    print(f"  epoch {epoch} step {step} loss {loss.item():.4f} {time.time()-started:.0f}s", flush=True)
            if stop:
                break
        model.eval()
        print("encoding dev/test", flush=True)
        neu_dev, neu_dev_s, neu_dev_ids = neural_rank(model, tokenizer, dev_q, dev_c, device, args.windows, args.width, args.pool_batch)
        neu_test, neu_test_s, neu_test_ids = neural_rank(model, tokenizer, test_q, test_c, device, args.windows, args.width, args.pool_batch)
        neu_report = micro_at_k(dev_q, neu_dev)
        print("neural dev", json.dumps(neu_report), flush=True)
        if args.rerank_top:
            ranked_dev = rerank_head(lex_dev, lex_dev_s, lex_dev_ids, neu_dev_s, neu_dev_ids, dev_q, args.rerank_top, args.rerank_mix)
            ranked_test = rerank_head(lex_test, lex_test_s, lex_test_ids, neu_test_s, neu_test_ids, test_q, args.rerank_top, args.rerank_mix)
            print("rerank top", args.rerank_top, "mix", args.rerank_mix, json.dumps(micro_at_k(dev_q, ranked_dev)), flush=True)
        elif args.mode == "blend":
            ranked_dev = blend_rank(lex_dev, neu_dev, dev_q, args.lex_w)
            ranked_test = blend_rank(lex_test, neu_test, test_q, args.lex_w)
        else:
            ranked_dev, ranked_test = neu_dev, neu_test
        torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    report = micro_at_k(dev_q, ranked_dev)
    print("dev report", json.dumps(report), flush=True)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"mode": args.mode, "arch": args.arch, "method": args.method, **report}, handle, indent=2)
    with open(os.path.join(args.out, "ranked_dev.json"), "w", encoding="utf-8") as handle:
        json.dump(cut(ranked_dev, POOL_K), handle)
    with open(os.path.join(args.out, "prediction_dev.json"), "w", encoding="utf-8") as handle:
        json.dump(cut(ranked_dev), handle)
    write_zip(os.path.join(args.out, "submission.zip"), ranked_test)


if __name__ == "__main__":
    main()
