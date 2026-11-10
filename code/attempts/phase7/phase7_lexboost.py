"""CPU lexical boosters for Phase 7. No GPU, no public checkpoints.

Sweeps TF-IDF / BM25 / head-field / citation overlap on dev, then writes a test zip.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = r"C:\Users\Anshuman Raj\CS781\phase7data\Phase_7_PCR"
OUT = r"C:\Users\Anshuman Raj\CS781\phase7_runs\lexboost"
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\lexboost"
K = 7
CITE = re.compile(
    r"(?:\b(?:air|scc|scr|supl|suppl)\b\s*\d*|\b(?:section|art(?:icle)?|s)\.?\s*\d+[A-Za-z0-9().-]*"
    r"|\b\d{4}\s+(?:air|scc|scr)\b|\b[A-Za-z][A-Za-z.&' -]{2,60}\bact\b(?:\s*,?\s*\d{4})?)",
    re.I,
)


def join_text(value):
    if isinstance(value, dict):
        value = value.get("text") or ""
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return str(value or "")


def clip_words(text, cap, mode="ends"):
    words = text.split()
    if cap <= 0 or len(words) <= cap:
        return text
    if mode == "head":
        return " ".join(words[:cap])
    if mode == "tail":
        return " ".join(words[-cap:])
    head = cap * 2 // 3
    return " ".join(words[:head] + words[-(cap - head):])


def load_split(kind):
    folder = {"train": "Train", "dev": "Dev", "test": "Test"}[kind]
    prefix = kind
    with open(os.path.join(ROOT, folder, f"{prefix}.json"), encoding="utf-8") as handle:
        queries = json.load(handle)
    with open(os.path.join(ROOT, folder, f"{prefix}_candidates.json"), encoding="utf-8") as handle:
        cands = json.load(handle)
    rows = []
    for key, value in queries.items():
        rel = []
        if isinstance(value, dict):
            rel = [str(item) for item in value.get("relevant_candidates") or [] if str(item).strip()]
        rows.append({"id": str(key), "text": join_text(value), "rel": rel})
    pool = {str(key): join_text(value) for key, value in cands.items()}
    return rows, pool


def micro_at_k(queries, ranked, k=K):
    tp = fp = fn = 0
    for row in queries:
        gold, pred = set(row["rel"]), set(ranked[row["id"]][:k])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - gold & pred) if False else len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return f1, precision, recall, tp, fp, fn


def rank_from_scores(queries, cand_ids, scores, k=50):
    ranked = {}
    for row, row_scores in zip(queries, scores):
        order = np.argsort(-row_scores)
        out = []
        qid = row["id"]
        for index in order:
            cid = cand_ids[index]
            if cid == qid:
                continue
            out.append(cid)
            if len(out) >= k:
                break
        ranked[qid] = out
    return ranked


def tfidf_scores(qtexts, docs, ngram=2, features=250000, analyzer="word", min_df=2, cap=4000, mode="ends"):
    q_clip = [clip_words(text, cap, mode) for text in qtexts]
    d_clip = [clip_words(text, cap, mode) for text in docs]
    kwargs = dict(
        ngram_range=(1, ngram) if analyzer == "word" else (3, 5),
        max_features=features,
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b",
        analyzer=analyzer,
    )
    if analyzer != "word":
        kwargs.pop("token_pattern")
        kwargs["ngram_range"] = (3, 5)
        kwargs["analyzer"] = "char_wb"
        kwargs["min_df"] = max(min_df, 3)
    vectorizer = TfidfVectorizer(**kwargs)
    matrix = normalize(vectorizer.fit_transform(d_clip + q_clip))
    return (matrix[len(docs):] @ matrix[:len(docs)].T).toarray()


def bm25_scores(qtexts, docs, k1=1.5, b=0.75, ngram=2, features=250000, cap=4000, mode="ends"):
    q_clip = [clip_words(text, cap, mode) for text in qtexts]
    d_clip = [clip_words(text, cap, mode) for text in docs]
    vectorizer = CountVectorizer(
        ngram_range=(1, ngram),
        max_features=features,
        min_df=2,
        max_df=0.95,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )
    doc_x = vectorizer.fit_transform(d_clip)
    query_x = vectorizer.transform(q_clip)
    df = np.asarray((doc_x > 0).sum(axis=0)).ravel()
    n_docs = doc_x.shape[0]
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    dl = np.asarray(doc_x.sum(axis=1)).ravel()
    avgdl = float(dl.mean()) or 1.0
    denom = k1 * (1.0 - b + b * (dl / avgdl))
    doc_x = doc_x.tocsr()
    indptr = doc_x.indptr
    weighted = doc_x.copy()
    for row in range(n_docs):
        start, end = indptr[row], indptr[row + 1]
        tf = doc_x.data[start:end]
        weighted.data[start:end] = idf[doc_x.indices[start:end]] * tf * (k1 + 1.0) / (tf + denom[row])
    return (query_x @ weighted.T).toarray()


def cite_scores(qtexts, docs):
    q_sets = [set(m.lower() for m in CITE.findall(text)) for text in qtexts]
    d_sets = [set(m.lower() for m in CITE.findall(text)) for text in docs]
    out = np.zeros((len(qtexts), len(docs)), dtype=np.float32)
    for i, qset in enumerate(q_sets):
        if not qset:
            continue
        for j, dset in enumerate(d_sets):
            if dset:
                inter = len(qset & dset)
                if inter:
                    out[i, j] = inter / np.sqrt(len(qset) * len(dset))
    return out


def write_zip(path, ranked):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pred = {qid: ids[:K] for qid, ids in ranked.items()}
    pred_path = os.path.join(os.path.dirname(path), "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(pred, handle)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("zip", path, "queries", len(pred), flush=True)


def eval_print(name, queries, cand_ids, scores):
    ranked = rank_from_scores(queries, cand_ids, scores)
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:28s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1, ranked, scores


def zscore(mat):
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return (mat - mean) / std


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    print("loading dev", flush=True)
    dev_q, dev_c = load_split("dev")
    cand_ids = list(dev_c)
    docs = [dev_c[cid] for cid in cand_ids]
    qtexts = [row["text"] for row in dev_q]
    print("dev", len(dev_q), len(docs), flush=True)

    variants = {}
    specs = [
        ("tfidf_w2_3k", dict(ngram=2, cap=3000, mode="ends", features=200000)),
        ("tfidf_w2_5k", dict(ngram=2, cap=5000, mode="ends", features=250000)),
        ("tfidf_w3_5k", dict(ngram=3, cap=5000, mode="ends", features=300000)),
        ("tfidf_head1200", dict(ngram=2, cap=1200, mode="head", features=150000)),
        ("tfidf_head800", dict(ngram=2, cap=800, mode="head", features=120000)),
        ("tfidf_char_head", dict(ngram=2, cap=1500, mode="head", features=180000, analyzer="char_wb")),
        ("bm25_w2_5k", None),
        ("bm25_head1200", None),
        ("cite", None),
    ]
    for name, spec in specs:
        print("scoring", name, flush=True)
        if name.startswith("tfidf"):
            scores = tfidf_scores(qtexts, docs, **spec)
        elif name == "bm25_w2_5k":
            scores = bm25_scores(qtexts, docs, cap=5000, mode="ends")
        elif name == "bm25_head1200":
            scores = bm25_scores(qtexts, docs, cap=1200, mode="head")
        else:
            scores = cite_scores(qtexts, docs)
        f1, ranked, scores = eval_print(name, dev_q, cand_ids, scores)
        variants[name] = {"f1": f1, "scores": scores, "ranked": ranked}

    print("blending", flush=True)
    keys = list(variants)
    zs = {name: zscore(variants[name]["scores"]) for name in keys}
    best = {"f1": -1, "weights": None, "name": None, "scores": None}
    # greedy: start from best single, add others
    order = sorted(keys, key=lambda name: variants[name]["f1"], reverse=True)
    current = zs[order[0]].copy()
    weights = {order[0]: 1.0}
    f1, _, _ = eval_print("start " + order[0], dev_q, cand_ids, current)
    best = {"f1": f1, "weights": dict(weights), "scores": current.copy()}
    for name in order[1:]:
        trial_best = None
        for coef in (0.15, 0.25, 0.4, 0.6, 0.9, 1.2):
            trial = current + coef * zs[name]
            f1, _, _ = eval_print(f" +{coef} {name}", dev_q, cand_ids, trial)
            if f1 > (trial_best[0] if trial_best else -1):
                trial_best = (f1, coef, trial)
        if trial_best and trial_best[0] > best["f1"] + 0.0005:
            current = trial_best[2]
            weights[name] = trial_best[1]
            best = {"f1": trial_best[0], "weights": dict(weights), "scores": current.copy()}
            print("keep", name, trial_best[1], "F1", trial_best[0], flush=True)
        else:
            print("skip", name, flush=True)

    print("BEST DEV", best["f1"], best["weights"], flush=True)
    with open(os.path.join(OUT, "blend.json"), "w", encoding="utf-8") as handle:
        json.dump({"f1": best["f1"], "weights": best["weights"],
                   "singles": {name: variants[name]["f1"] for name in keys}}, handle, indent=2)

    print("loading test", flush=True)
    test_q, test_c = load_split("test")
    t_ids = list(test_c)
    t_docs = [test_c[cid] for cid in t_ids]
    t_q = [row["text"] for row in test_q]
    test_scores = None
    for name, coef in best["weights"].items():
        print("test scoring", name, coef, flush=True)
        if name.startswith("tfidf"):
            spec = dict(specs[[item[0] for item in specs].index(name)][1])
            part = tfidf_scores(t_q, t_docs, **spec)
        elif name == "bm25_w2_5k":
            part = bm25_scores(t_q, t_docs, cap=5000, mode="ends")
        elif name == "bm25_head1200":
            part = bm25_scores(t_q, t_docs, cap=1200, mode="head")
        elif name == "cite":
            part = cite_scores(t_q, t_docs)
        else:
            continue
        part = zscore(part)
        test_scores = part * coef if test_scores is None else test_scores + part * coef
    ranked = rank_from_scores(test_q, t_ids, test_scores)
    write_zip(os.path.join(OUT, "submission.zip"), ranked)
    write_zip(os.path.join(SUB, "submission.zip"), ranked)
    with open(os.path.join(OUT, "prediction_dev.json"), "w", encoding="utf-8") as handle:
        json.dump({qid: ids[:K] for qid, ids in rank_from_scores(dev_q, cand_ids, best["scores"]).items()}, handle)


if __name__ == "__main__":
    main()
