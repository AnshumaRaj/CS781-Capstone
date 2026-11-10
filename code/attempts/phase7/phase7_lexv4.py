"""Beat 0.4182 without overfitting the 118-query dev set.

Tune on train (827 queries), confirm on dev, then write a test zip.
No GPU. No public checkpoints.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from phase7_lexboost import (
    CITE, K, clip_words, load_split, micro_at_k, rank_from_scores, write_zip, zscore, bm25_scores,
)

OUT = r"C:\Users\Anshuman Raj\CS781\phase7_runs\lexv4"
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\lexv4"
MASK = re.compile(r"<[^>]+>")
HEAD = re.compile(
    r"(?:CITATION|HEADNOTE|ACT|PETITIONER|RESPONDENT|BENCH)\s*:(.{0,2500})",
    re.I | re.S,
)


def strip_mask(text):
    return MASK.sub(" ", text)


def field_boost(text):
    bits = [m.group(0) for m in HEAD.finditer(text)]
    extra = " ".join(bits) if bits else " ".join(text.split()[:400])
    return clip_words(text, 8000) + " " + extra + " " + extra


def show(name, queries, ids, scores):
    ranked = rank_from_scores(queries, ids, scores)
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:32s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1, ranked, scores


def make_vec(ngram=3, features=400000, min_df=2, max_df=0.95, sublinear=True, ngram_range=None):
    return TfidfVectorizer(
        ngram_range=ngram_range or (1, ngram),
        max_features=features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear,
        lowercase=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b",
    )


def cosine(vec, qtexts, docs):
    matrix = normalize(vec.transform(list(docs) + list(qtexts)))
    return (matrix[len(docs):] @ matrix[:len(docs)].T).toarray()


def fit_pooled(texts, **kwargs):
    vec = make_vec(**kwargs)
    vec.fit(texts)
    print(" vocab", len(vec.vocabulary_), "docs", len(texts), flush=True)
    return vec


def cite_tfidf(qtexts, docs):
    q = [" ".join(CITE.findall(text)) for text in qtexts]
    d = [" ".join(CITE.findall(text)) for text in docs]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True, dtype=np.float32)
    matrix = normalize(vec.fit_transform(d + q))
    return (matrix[len(docs):] @ matrix[:len(docs)].T).toarray()


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    print("load", flush=True)
    train_q, train_c = load_split("train")
    dev_q, dev_c = load_split("dev")
    test_q, test_c = load_split("test")
    print("sizes", len(train_q), len(train_c), len(dev_q), len(dev_c), len(test_q), len(test_c), flush=True)

    train_ids, dev_ids = list(train_c), list(dev_c)
    train_docs_raw = [train_c[cid] for cid in train_ids]
    dev_docs_raw = [dev_c[cid] for cid in dev_ids]
    train_q_raw = [row["text"] for row in train_q]
    dev_q_raw = [row["text"] for row in dev_q]

    pooled_8k = [clip_words(t, 8000) for t in list(train_c.values()) + list(dev_c.values()) + list(test_c.values())]
    print("fit pooled 8k", len(pooled_8k), flush=True)
    vec8 = fit_pooled(pooled_8k, ngram=3, features=400000)
    vec8b = fit_pooled(pooled_8k, ngram=3, features=600000, max_df=0.9)
    vec23 = fit_pooled(pooled_8k, ngram=3, features=400000, ngram_range=(2, 3))
    vec8strip_src = [clip_words(strip_mask(t), 8000) for t in list(train_c.values()) + list(dev_c.values()) + list(test_c.values())]
    vec8s = fit_pooled(vec8strip_src, ngram=3, features=400000)

    variants = {}

    def add(split, name, scores, queries, ids):
        f1, ranked, scores = show(f"{split}/{name}", queries, ids, scores)
        variants.setdefault(name, {})[split] = {"f1": f1, "scores": scores, "ranked": ranked, "ids": ids}
        return f1

    print("=== train/dev singles ===", flush=True)
    plans = [
        ("base8k_local", None),
        ("pooled8k", vec8),
        ("pooled8k_600k", vec8b),
        ("pooled23", vec23),
        ("pooled_strip", vec8s),
        ("qboost", vec8),
    ]
    # local 8k baseline (fit on split only) — the 0.4182 recipe
    tr_q8 = [clip_words(t, 8000) for t in train_q_raw]
    tr_d8 = [clip_words(t, 8000) for t in train_docs_raw]
    dv_q8 = [clip_words(t, 8000) for t in dev_q_raw]
    dv_d8 = [clip_words(t, 8000) for t in dev_docs_raw]
    local_tr = make_vec(3, 400000)
    local_tr.fit(tr_d8)
    add("train", "base8k_local", cosine(local_tr, tr_q8, tr_d8), train_q, train_ids)
    local_dv = make_vec(3, 400000)
    local_dv.fit(dv_d8)
    add("dev", "base8k_local", cosine(local_dv, dv_q8, dv_d8), dev_q, dev_ids)

    add("train", "pooled8k", cosine(vec8, tr_q8, tr_d8), train_q, train_ids)
    add("dev", "pooled8k", cosine(vec8, dv_q8, dv_d8), dev_q, dev_ids)
    add("train", "pooled8k_600k", cosine(vec8b, tr_q8, tr_d8), train_q, train_ids)
    add("dev", "pooled8k_600k", cosine(vec8b, dv_q8, dv_d8), dev_q, dev_ids)
    add("train", "pooled23", cosine(vec23, tr_q8, tr_d8), train_q, train_ids)
    add("dev", "pooled23", cosine(vec23, dv_q8, dv_d8), dev_q, dev_ids)

    tr_qs = [clip_words(strip_mask(t), 8000) for t in train_q_raw]
    tr_ds = [clip_words(strip_mask(t), 8000) for t in train_docs_raw]
    dv_qs = [clip_words(strip_mask(t), 8000) for t in dev_q_raw]
    dv_ds = [clip_words(strip_mask(t), 8000) for t in dev_docs_raw]
    add("train", "pooled_strip", cosine(vec8s, tr_qs, tr_ds), train_q, train_ids)
    add("dev", "pooled_strip", cosine(vec8s, dv_qs, dv_ds), dev_q, dev_ids)

    tr_qb = [field_boost(t) for t in train_q_raw]
    dv_qb = [field_boost(t) for t in dev_q_raw]
    add("train", "qboost", cosine(vec8, tr_qb, tr_d8), train_q, train_ids)
    add("dev", "qboost", cosine(vec8, dv_qb, dv_d8), dev_q, dev_ids)

    print("bm25", flush=True)
    for k1, b in ((1.2, 0.6), (0.9, 0.4), (1.5, 0.75)):
        name = f"bm25_{k1}_{b}"
        add("train", name, bm25_scores(train_q_raw, train_docs_raw, k1=k1, b=b, ngram=3, features=400000, cap=8000), train_q, train_ids)
        add("dev", name, bm25_scores(dev_q_raw, dev_docs_raw, k1=k1, b=b, ngram=3, features=400000, cap=8000), dev_q, dev_ids)

    print("cite", flush=True)
    add("train", "cite", cite_tfidf(train_q_raw, train_docs_raw), train_q, train_ids)
    add("dev", "cite", cite_tfidf(dev_q_raw, dev_docs_raw), dev_q, dev_ids)

    print("7k/9k pooled", flush=True)
    pooled_7k = [clip_words(t, 7000) for t in list(train_c.values()) + list(dev_c.values()) + list(test_c.values())]
    pooled_9k = [clip_words(t, 9000) for t in list(train_c.values()) + list(dev_c.values()) + list(test_c.values())]
    vec7 = fit_pooled(pooled_7k, ngram=3, features=400000)
    vec9 = fit_pooled(pooled_9k, ngram=3, features=450000)
    add("train", "pooled7k", cosine(vec7, [clip_words(t, 7000) for t in train_q_raw], [clip_words(t, 7000) for t in train_docs_raw]), train_q, train_ids)
    add("dev", "pooled7k", cosine(vec7, [clip_words(t, 7000) for t in dev_q_raw], [clip_words(t, 7000) for t in dev_docs_raw]), dev_q, dev_ids)
    add("train", "pooled9k", cosine(vec9, [clip_words(t, 9000) for t in train_q_raw], [clip_words(t, 9000) for t in train_docs_raw]), train_q, train_ids)
    add("dev", "pooled9k", cosine(vec9, [clip_words(t, 9000) for t in dev_q_raw], [clip_words(t, 9000) for t in dev_docs_raw]), dev_q, dev_ids)

    print("=== summary ===", flush=True)
    names = sorted({name for name in variants}, key=lambda n: variants[n].get("train", {}).get("f1", 0), reverse=True)
    for name in names:
        tr = variants[name].get("train", {}).get("f1")
        dv = variants[name].get("dev", {}).get("f1")
        print(f"  {name:24s} train={tr:.4f}  dev={dv:.4f}" if tr is not None and dv is not None else name, flush=True)

    print("blend on TRAIN", flush=True)
    # start from best train single that also has a score matrix
    start = names[0]
    current_tr = zscore(variants[start]["train"]["scores"])
    current_dv = zscore(variants[start]["dev"]["scores"])
    weights = {start: 1.0}
    best_tr = variants[start]["train"]["f1"]
    best_dv = variants[start]["dev"]["f1"]
    for name in names[1:]:
        if "train" not in variants[name] or "dev" not in variants[name]:
            continue
        trial_best = None
        for coef in (0.1, 0.2, 0.35, 0.5, 0.75):
            trial_tr = current_tr + coef * zscore(variants[name]["train"]["scores"])
            f1_tr = show(f"tr +{coef} {name}", train_q, train_ids, trial_tr)[0]
            if f1_tr > (trial_best[0] if trial_best else -1):
                trial_dv = current_dv + coef * zscore(variants[name]["dev"]["scores"])
                f1_dv = show(f"dv +{coef} {name}", dev_q, dev_ids, trial_dv)[0]
                trial_best = (f1_tr, f1_dv, coef, trial_tr, trial_dv)
        # keep if train improves and dev does not collapse below baseline 0.430 by much
        if trial_best and trial_best[0] > best_tr + 0.0008 and trial_best[1] >= min(best_dv, 0.425) - 0.002:
            current_tr, current_dv = trial_best[3], trial_best[4]
            weights[name] = trial_best[2]
            best_tr, best_dv = trial_best[0], trial_best[1]
            print("KEEP", name, trial_best[2], "train", best_tr, "dev", best_dv, flush=True)
        else:
            print("skip", name, flush=True)

    print("CHOSEN", weights, "train", best_tr, "dev", best_dv, flush=True)
    with open(os.path.join(OUT, "choice.json"), "w") as handle:
        json.dump({"weights": weights, "train": best_tr, "dev": best_dv,
                   "singles": {n: {"train": variants[n]["train"]["f1"], "dev": variants[n]["dev"]["f1"]} for n in names if "dev" in variants[n]}}, handle, indent=2)

    print("test", flush=True)
    test_ids = list(test_c)
    test_docs_raw = [test_c[cid] for cid in test_ids]
    test_q_raw = [row["text"] for row in test_q]
    te_d8 = [clip_words(t, 8000) for t in test_docs_raw]
    te_q8 = [clip_words(t, 8000) for t in test_q_raw]
    test_parts = {}
    makers = {
        "base8k_local": None,
        "pooled8k": lambda: cosine(vec8, te_q8, te_d8),
        "pooled8k_600k": lambda: cosine(vec8b, te_q8, te_d8),
        "pooled23": lambda: cosine(vec23, te_q8, te_d8),
        "pooled_strip": lambda: cosine(vec8s, [clip_words(strip_mask(t), 8000) for t in test_q_raw], [clip_words(strip_mask(t), 8000) for t in test_docs_raw]),
        "qboost": lambda: cosine(vec8, [field_boost(t) for t in test_q_raw], te_d8),
        "pooled7k": lambda: cosine(vec7, [clip_words(t, 7000) for t in test_q_raw], [clip_words(t, 7000) for t in test_docs_raw]),
        "pooled9k": lambda: cosine(vec9, [clip_words(t, 9000) for t in test_q_raw], [clip_words(t, 9000) for t in test_docs_raw]),
        "cite": lambda: cite_tfidf(test_q_raw, test_docs_raw),
    }
    for k1, b in ((1.2, 0.6), (0.9, 0.4), (1.5, 0.75)):
        makers[f"bm25_{k1}_{b}"] = (lambda k1=k1, b=b: bm25_scores(test_q_raw, test_docs_raw, k1=k1, b=b, ngram=3, features=400000, cap=8000))

    # fix base8k_local maker
    def local8k_test():
        vec = make_vec(3, 400000)
        vec.fit(te_d8)
        return cosine(vec, te_q8, te_d8)
    makers["base8k_local"] = local8k_test

    test_scores = None
    for name, coef in weights.items():
        print("test part", name, coef, flush=True)
        part = zscore(makers[name]())
        test_scores = part * coef if test_scores is None else test_scores + part * coef
    ranked = rank_from_scores(test_q, test_ids, test_scores)
    write_zip(os.path.join(OUT, "submission.zip"), ranked)
    write_zip(os.path.join(SUB, "submission.zip"), ranked)
    with open(os.path.join(OUT, "prediction_dev.json"), "w") as handle:
        json.dump({qid: ids[:K] for qid, ids in rank_from_scores(dev_q, dev_ids, current_dv).items()}, handle)
    print("done train", best_tr, "dev", best_dv, flush=True)


if __name__ == "__main__":
    main()
