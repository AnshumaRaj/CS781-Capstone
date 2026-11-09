"""Rerank the winning 8k TF-IDF list. Fit vectorizer on each split's docs+queries.

CPU only. Writes a test zip if dev F1 beats 0.4302.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from phase7_lexboost import CITE, K, clip_words, load_split, micro_at_k, rank_from_scores, write_zip, zscore
from phase7_lexv4 import cite_tfidf, field_boost

OUT = r"C:\Users\Anshuman Raj\CS781\phase7_runs\rerank"
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\rerank"
BASE_DEV = 0.430239


def vec_fit(qtexts, docs, ngram=3, features=400000, cap=8000):
    q_clip = [clip_words(t, cap) for t in qtexts]
    d_clip = [clip_words(t, cap) for t in docs]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram), max_features=features, min_df=2, max_df=0.95,
        sublinear_tf=True, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b",
    )
    matrix = normalize(vectorizer.fit_transform(d_clip + q_clip))
    return (matrix[len(docs):] @ matrix[:len(docs)].T).toarray()


def head_fit(qtexts, docs, cap=1200):
    q_clip = [clip_words(t, cap, "head") for t in qtexts]
    d_clip = [clip_words(t, cap, "head") for t in docs]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), max_features=200000, min_df=2, max_df=0.95,
        sublinear_tf=True, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b",
    )
    matrix = normalize(vectorizer.fit_transform(d_clip + q_clip))
    return (matrix[len(docs):] @ matrix[:len(docs)].T).toarray()


def show(name, queries, ids, scores):
    ranked = rank_from_scores(queries, ids, scores)
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:36s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1, ranked, scores


def blend(a, b, coef):
    return zscore(a) + coef * zscore(b)


def ltr_rerank(queries, ids, base, extra, topn=25, C=1.0):
    """Train logistic on train split extras, apply as additive rerank of topn."""
    # extra: list of score matrices aligned with base
    n_f = 1 + len(extra)
    X, y = [], []
    for i, row in enumerate(queries):
        gold = set(row["rel"])
        if not gold:
            continue
        order = np.argsort(-base[i])
        pos = [j for j in order if ids[j] in gold][:12]
        neg = [j for j in order if ids[j] not in gold and ids[j] != row["id"]][:18]
        for j, label in [(p, 1) for p in pos] + [(n, 0) for n in neg]:
            feats = [base[i, j]] + [mat[i, j] for mat in extra]
            X.append(feats)
            y.append(label)
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
    clf = LogisticRegression(C=C, max_iter=200, class_weight="balanced")
    clf.fit(X, y)
    print("ltr coef", clf.coef_, "bias", float(clf.intercept_[0]), "n", len(y), "pos", int(y.sum()), flush=True)
    out = base.copy()
    for i, row in enumerate(queries):
        order = np.argsort(-base[i])[:topn]
        feats = np.stack([base[i, order]] + [mat[i, order] for mat in extra], 1)
        boost = clf.decision_function(feats)
        # keep mass on original ranking, add scaled boost
        new = base[i].copy()
        new[order] = zscore(base[i:i+1, order])[0] + 0.35 * zscore(boost.reshape(1, -1))[0]
        out[i] = new
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    print("load", flush=True)
    train_q, train_c = load_split("train")
    dev_q, dev_c = load_split("dev")
    test_q, test_c = load_split("test")
    tr_ids, dv_ids, te_ids = list(train_c), list(dev_c), list(test_c)
    tr_docs = [train_c[c] for c in tr_ids]
    dv_docs = [dev_c[c] for c in dv_ids]
    te_docs = [test_c[c] for c in te_ids]
    tr_q = [r["text"] for r in train_q]
    dv_q = [r["text"] for r in dev_q]
    te_q = [r["text"] for r in test_q]

    print("8k base", flush=True)
    tr_base = vec_fit(tr_q, tr_docs)
    dv_base = vec_fit(dv_q, dv_docs)
    te_base = vec_fit(te_q, te_docs)
    show("train/8k", train_q, tr_ids, tr_base)
    f1_8k, _, _ = show("dev/8k", dev_q, dv_ids, dv_base)

    print("aux", flush=True)
    tr_cite, dv_cite, te_cite = cite_tfidf(tr_q, tr_docs), cite_tfidf(dv_q, dv_docs), cite_tfidf(te_q, te_docs)
    tr_head, dv_head, te_head = head_fit(tr_q, tr_docs), head_fit(dv_q, dv_docs), head_fit(te_q, te_docs)
    show("dev/cite", dev_q, dv_ids, dv_cite)
    show("dev/head", dev_q, dv_ids, dv_head)

    print("grid blend on DEV (confirm on train)", flush=True)
    best = {"f1": f1_8k, "name": "8k", "dev": dv_base, "test": te_base}
    for name, tr_m, dv_m, te_m in (("cite", tr_cite, dv_cite, te_cite), ("head", tr_head, dv_head, te_head)):
        for coef in (0.05, 0.08, 0.12, 0.18, 0.25, 0.35):
            dv = blend(dv_base, dv_m, coef)
            f1, _, _ = show(f"dev/8k+{coef}{name}", dev_q, dv_ids, dv)
            if f1 > best["f1"] + 0.0004:
                tr = blend(tr_base, tr_m, coef)
                f1_tr, _, _ = show(f"train/8k+{coef}{name}", train_q, tr_ids, tr)
                best = {"f1": f1, "name": f"8k+{coef}{name}", "dev": dv, "test": blend(te_base, te_m, coef), "train": f1_tr}
                print(" keep", best["name"], "dev", f1, "train", f1_tr, flush=True)

    # both cite and head
    for c_cite in (0.08, 0.12, 0.18):
        for c_head in (0.05, 0.1, 0.18):
            dv = zscore(dv_base) + c_cite * zscore(dv_cite) + c_head * zscore(dv_head)
            f1, _, _ = show(f"dev/8k+{c_cite}c+{c_head}h", dev_q, dv_ids, dv)
            if f1 > best["f1"] + 0.0004:
                tr = zscore(tr_base) + c_cite * zscore(tr_cite) + c_head * zscore(tr_head)
                f1_tr, _, _ = show(f"train/8k+{c_cite}c+{c_head}h", train_q, tr_ids, tr)
                te = zscore(te_base) + c_cite * zscore(te_cite) + c_head * zscore(te_head)
                best = {"f1": f1, "name": f"8k+{c_cite}c+{c_head}h", "dev": dv, "test": te, "train": f1_tr}
                print(" keep", best["name"], "dev", f1, "train", f1_tr, flush=True)

    print("LTR rerank of 8k top-25 using cite+head", flush=True)
    # train LTR on train, apply to all
    def train_ltr(base, cite, head, queries, ids):
        X, y = [], []
        for i, row in enumerate(queries):
            gold = set(row["rel"])
            if not gold:
                continue
            order = np.argsort(-base[i])
            pos = [j for j in order if ids[j] in gold][:12]
            neg = [j for j in order if ids[j] not in gold and ids[j] != row["id"]][:18]
            bz, cz, hz = zscore(base[i:i+1])[0], zscore(cite[i:i+1])[0], zscore(head[i:i+1])[0]
            for j in pos:
                X.append([bz[j], cz[j], hz[j]])
                y.append(1)
            for j in neg:
                X.append([bz[j], cz[j], hz[j]])
                y.append(0)
        clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")
        clf.fit(np.asarray(X, np.float32), np.asarray(y))
        print("ltr w", clf.coef_, flush=True)
        return clf

    clf = train_ltr(tr_base, tr_cite, tr_head, train_q, tr_ids)

    def apply_ltr(base, cite, head, topn=25, mix=0.4):
        out = base.copy()
        for i in range(base.shape[0]):
            order = np.argsort(-base[i])[:topn]
            bz, cz, hz = zscore(base[i:i+1])[0], zscore(cite[i:i+1])[0], zscore(head[i:i+1])[0]
            feats = np.stack([bz[order], cz[order], hz[order]], 1)
            boost = clf.decision_function(feats)
            fused = zscore(base[i:i+1, order])[0] + mix * zscore(boost.reshape(1, -1))[0]
            out[i, order] = fused
        return out

    for mix in (0.2, 0.35, 0.5, 0.7):
        for topn in (15, 25, 40):
            dv = apply_ltr(dv_base, dv_cite, dv_head, topn=topn, mix=mix)
            f1, _, _ = show(f"dev/ltr top{topn} mix{mix}", dev_q, dv_ids, dv)
            if f1 > best["f1"] + 0.0004:
                tr = apply_ltr(tr_base, tr_cite, tr_head, topn=topn, mix=mix)
                f1_tr, _, _ = show(f"train/ltr top{topn} mix{mix}", train_q, tr_ids, tr)
                te = apply_ltr(te_base, te_cite, te_head, topn=topn, mix=mix)
                best = {"f1": f1, "name": f"ltr-{topn}-{mix}", "dev": dv, "test": te, "train": f1_tr}
                print(" keep", best["name"], "dev", f1, "train", f1_tr, flush=True)

    print("BEST", best["name"], "dev", best["f1"], "train", best.get("train"), flush=True)
    with open(os.path.join(OUT, "choice.json"), "w") as handle:
        json.dump({"name": best["name"], "dev": best["f1"], "train": best.get("train"), "base_dev": f1_8k}, handle, indent=2)

    ranked = rank_from_scores(test_q, te_ids, best["test"])
    write_zip(os.path.join(OUT, "submission.zip"), ranked)
    write_zip(os.path.join(SUB, "submission.zip"), ranked)
    with open(os.path.join(OUT, "prediction_dev.json"), "w") as handle:
        json.dump({qid: ids[:K] for qid, ids in rank_from_scores(dev_q, dv_ids, best["dev"]).items()}, handle)
    print("wrote", best["name"], best["f1"], "vs base", f1_8k, flush=True)


if __name__ == "__main__":
    main()
