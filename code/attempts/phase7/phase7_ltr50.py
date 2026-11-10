"""Reorder only the 8k top-50. Do not mix score scales with the rest of the pool."""
from __future__ import annotations

import json
import os
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from phase7_lexboost import K, load_split, micro_at_k, rank_from_scores, write_zip
from phase7_rerank import vec_fit, head_fit, cite_tfidf

OUT = r"C:\Users\Anshuman Raj\CS781\phase7_runs\ltr50"
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\ltr50"
AIR = re.compile(r"\b(?:19|20)\d{2}\s+AIR\s+\d+\b", re.I)
SCC = re.compile(r"\b(?:19|20)\d{2}\s+\d+\s+SCC\s+\d+\b", re.I)
SEC = re.compile(r"\b(?:section|art(?:icle)?|s)\.?\s*\d+[A-Za-z0-9()]*", re.I)
ACT = re.compile(r"\b[A-Za-z][A-Za-z.&' -]{2,40}\sAct(?:,?\s*\d{4})?", re.I)


def cite_set(text):
    blob = " ".join(text.split()[:2500])
    out = set()
    for pat in (AIR, SCC, SEC, ACT):
        out.update(m.group(0).lower() for m in pat.finditer(blob))
    return out


def show(name, queries, ranked):
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:32s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1


def feats(i, j, base, head, cite, ids, qid, qcite, dcite):
    rank = None
    return [
        float(base[i, j]),
        float(head[i, j]),
        float(cite[i, j]),
        float(len(qcite & dcite)),
        1.0 if ids[j] == qid else 0.0,
    ]


def reorder(queries, ids, base, head, cite, q_cites, d_cites, clf, scaler, topn=50):
    ranked = {}
    index = {c: n for n, c in enumerate(ids)}
    for i, row in enumerate(queries):
        order = [j for j in np.argsort(-base[i]) if ids[j] != row["id"]]
        head_idx = order[:topn]
        rest = order[topn:]
        X = []
        for rnk, j in enumerate(head_idx):
            X.append([
                float(base[i, j]),
                float(head[i, j]),
                float(cite[i, j]),
                float(len(q_cites[i] & d_cites[j])),
                1.0 / (rnk + 1),
            ])
        scores = clf.decision_function(scaler.transform(np.asarray(X, np.float32)))
        new_head = [head_idx[k] for k in np.argsort(-scores)]
        ranked[row["id"]] = [ids[j] for j in new_head + rest]
    return ranked


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    print("load", flush=True)
    train_q, train_c = load_split("train")
    dev_q, dev_c = load_split("dev")
    test_q, test_c = load_split("test")
    tr_ids, dv_ids, te_ids = list(train_c), list(dev_c), list(test_c)

    print("8k/head/cite", flush=True)
    tr_base = vec_fit([r["text"] for r in train_q], [train_c[c] for c in tr_ids])
    dv_base = vec_fit([r["text"] for r in dev_q], [dev_c[c] for c in dv_ids])
    te_base = vec_fit([r["text"] for r in test_q], [test_c[c] for c in te_ids])
    print("base", show("dev/8k", dev_q, rank_from_scores(dev_q, dv_ids, dv_base)), flush=True)

    tr_head = head_fit([r["text"] for r in train_q], [train_c[c] for c in tr_ids])
    dv_head = head_fit([r["text"] for r in dev_q], [dev_c[c] for c in dv_ids])
    te_head = head_fit([r["text"] for r in test_q], [test_c[c] for c in te_ids])
    tr_cite = cite_tfidf([r["text"] for r in train_q], [train_c[c] for c in tr_ids])
    dv_cite = cite_tfidf([r["text"] for r in dev_q], [dev_c[c] for c in dv_ids])
    te_cite = cite_tfidf([r["text"] for r in test_q], [test_c[c] for c in te_ids])

    print("structured cites", flush=True)
    tr_qc = [cite_set(r["text"]) for r in train_q]
    tr_dc = [cite_set(train_c[c]) for c in tr_ids]
    dv_qc = [cite_set(r["text"]) for r in dev_q]
    dv_dc = [cite_set(dev_c[c]) for c in dv_ids]
    te_qc = [cite_set(r["text"]) for r in test_q]
    te_dc = [cite_set(test_c[c]) for c in te_ids]

    print("train pairs from 8k top50", flush=True)
    X, y = [], []
    for i, row in enumerate(train_q):
        gold = set(row["rel"])
        if not gold:
            continue
        order = [j for j in np.argsort(-tr_base[i]) if tr_ids[j] != row["id"]][:50]
        for rnk, j in enumerate(order):
            X.append([
                float(tr_base[i, j]),
                float(tr_head[i, j]),
                float(tr_cite[i, j]),
                float(len(tr_qc[i] & tr_dc[j])),
                1.0 / (rnk + 1),
            ])
            y.append(1 if tr_ids[j] in gold else 0)
    X, y = np.asarray(X, np.float32), np.asarray(y)
    print("pairs", len(y), "pos", int(y.sum()), "rate", round(float(y.mean()), 3), flush=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    best = {"f1": show("dev/8k", dev_q, rank_from_scores(dev_q, dv_ids, dv_base)), "C": None}
    for C in (0.1, 0.3, 1.0, 3.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=400, class_weight="balanced")
        clf.fit(Xs, y)
        print("C", C, "w", np.round(clf.coef_[0], 3), flush=True)
        ranked = reorder(dev_q, dv_ids, dv_base, dv_head, dv_cite, dv_qc, dv_dc, clf, scaler, 50)
        f1 = show(f"dev/ltr50 C={C}", dev_q, ranked)
        if f1 > best["f1"] + 0.0005:
            best = {"f1": f1, "C": C, "clf": clf, "ranked": ranked}
            print(" KEEP", C, f1, flush=True)
        ranked20 = reorder(dev_q, dv_ids, dv_base, dv_head, dv_cite, dv_qc, dv_dc, clf, scaler, 20)
        f120 = show(f"dev/ltr20 C={C}", dev_q, ranked20)
        if f120 > best["f1"] + 0.0005:
            best = {"f1": f120, "C": C, "clf": clf, "ranked": ranked20, "topn": 20}
            print(" KEEP20", C, f120, flush=True)

    print("BEST", best.get("C"), best["f1"], flush=True)
    if best["C"] is None:
        print("no gain over 8k", flush=True)
        return
    topn = best.get("topn", 50)
    test_ranked = reorder(test_q, te_ids, te_base, te_head, te_cite, te_qc, te_dc, best["clf"], scaler, topn)
    write_zip(os.path.join(OUT, "submission.zip"), test_ranked)
    write_zip(os.path.join(SUB, "submission.zip"), test_ranked)
    with open(os.path.join(OUT, "prediction_dev.json"), "w") as handle:
        json.dump({qid: ids[:K] for qid, ids in best["ranked"].items()}, handle)
    with open(os.path.join(OUT, "choice.json"), "w") as handle:
        json.dump({"C": best["C"], "f1": best["f1"], "topn": topn, "w": best["clf"].coef_[0].tolist()}, handle, indent=2)


if __name__ == "__main__":
    main()
