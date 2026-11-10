"""Train-gold query expansion + top-N rerank. No public BERT. CPU only."""
from __future__ import annotations

import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from phase7_lexboost import K, clip_words, load_split, micro_at_k, rank_from_scores, write_zip, zscore
from phase7_lexv4 import cite_tfidf

OUT = r"C:\Users\Anshuman Raj\CS781\phase7_runs\expand"
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\expand"
SENT_SPLIT = re.compile(r"(?<=[.;])\s+")


def show(name, queries, ids, scores):
    ranked = rank_from_scores(queries, ids, scores)
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:40s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1, ranked, scores


def fit_space(texts, ngram=3, features=400000):
    vec = TfidfVectorizer(
        ngram_range=(1, ngram), max_features=features, min_df=2, max_df=0.95,
        sublinear_tf=True, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b",
    )
    vec.fit(texts)
    return vec


def embed(vec, texts):
    return normalize(vec.transform(texts))


def local_8k(queries, pool):
    ids = list(pool)
    q = [clip_words(r["text"], 8000) for r in queries]
    d = [clip_words(pool[c], 8000) for c in ids]
    vec = TfidfVectorizer(
        ngram_range=(1, 3), max_features=400000, min_df=2, max_df=0.95,
        sublinear_tf=True, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b",
    )
    matrix = normalize(vec.fit_transform(d + q))
    scores = (matrix[len(d):] @ matrix[:len(d)].T).toarray()
    return ids, scores, vec, matrix[len(d):], matrix[:len(d)]


def gold_centroids(queries, ids, doc_x):
    index = {c: i for i, c in enumerate(ids)}
    out = []
    for row in queries:
        js = [index[c] for c in row["rel"] if c in index]
        if js:
            out.append(np.asarray(doc_x[js].mean(axis=0)))
        else:
            out.append(np.zeros((1, doc_x.shape[1]), dtype=np.float32))
    return np.vstack(out)


def knn_expand(query_x, donor_q, donor_cent, k=5, alpha=0.25):
    # query_x: nq x v, donor_q: ndonor x v
    sim = (query_x @ donor_q.T).toarray() if hasattr(query_x, "toarray") else query_x @ donor_q.T
    if hasattr(sim, "toarray"):
        sim = sim.toarray()
    donor_q_arr = donor_q.toarray() if hasattr(donor_q, "toarray") else donor_q
    query_arr = query_x.toarray() if hasattr(query_x, "toarray") else query_x
    expanded = []
    for i in range(query_arr.shape[0]):
        order = np.argsort(-sim[i])[:k]
        # skip self if donor is the same set
        picked = []
        for j in np.argsort(-sim[i]):
            if donor_q is query_x and j == i:
                continue
            picked.append(j)
            if len(picked) >= k:
                break
        cent = donor_cent[picked].mean(axis=0)
        mixed = (1 - alpha) * query_arr[i] + alpha * cent
        nrm = np.linalg.norm(mixed) or 1.0
        expanded.append(mixed / nrm)
    return np.vstack(expanded).astype(np.float32)


def sent_max_for_top(queries, pool, ids, base, vec, q_x, topn=20, max_sents=40):
    out = np.zeros_like(base)
    id_to_text = pool
    for i, row in enumerate(queries):
        order = [j for j in np.argsort(-base[i]) if ids[j] != row["id"]][:topn]
        qv = q_x[i]
        rows = []
        owners = []
        for j in order:
            sents = [s.strip() for s in SENT_SPLIT.split(id_to_text[ids[j]]) if 6 <= len(s.split()) <= 80][:max_sents]
            if not sents:
                sents = [clip_words(id_to_text[ids[j]], 400, "head")]
            for s in sents:
                rows.append(s)
                owners.append(j)
        if not rows:
            continue
        sx = embed(vec, rows)
        sim = (sx @ qv.T).toarray().ravel() if hasattr(qv, "T") else np.asarray(sx @ qv.T).ravel()
        # qv is 1 x V sparse
        qv2 = qv.toarray() if hasattr(qv, "toarray") else np.asarray(qv).reshape(1, -1)
        sim = np.asarray(sx @ qv2.T).ravel()
        best = {}
        for owner, val in zip(owners, sim):
            if owner not in best or val > best[owner]:
                best[owner] = val
        for j, val in best.items():
            out[i, j] = val
    return out


def rerank_top(base, bonus, topn=20, mix=0.25):
    out = base.copy()
    for i in range(base.shape[0]):
        order = np.argsort(-base[i])[:topn]
        fused = zscore(base[i:i + 1, order])[0] + mix * zscore(bonus[i:i + 1, order])[0]
        out[i, order] = fused
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    print("load", flush=True)
    train_q, train_c = load_split("train")
    dev_q, dev_c = load_split("dev")
    test_q, test_c = load_split("test")

    print("local 8k", flush=True)
    tr_ids, tr_base, tr_vec, tr_qx, tr_dx = local_8k(train_q, train_c)
    dv_ids, dv_base, dv_vec, dv_qx, dv_dx = local_8k(dev_q, dev_c)
    te_ids, te_base, te_vec, te_qx, te_dx = local_8k(test_q, test_c)
    show("train/8k", train_q, tr_ids, tr_base)
    f1_8k, _, _ = show("dev/8k", dev_q, dv_ids, dv_base)

    print("pooled space for expansion", flush=True)
    pooled_texts = (
        [clip_words(train_c[c], 8000) for c in train_c]
        + [clip_words(dev_c[c], 8000) for c in dev_c]
        + [clip_words(test_c[c], 8000) for c in test_c]
        + [clip_words(r["text"], 8000) for r in train_q + dev_q + test_q]
    )
    pvec = fit_space(pooled_texts)
    tr_pq = embed(pvec, [clip_words(r["text"], 8000) for r in train_q])
    tr_pd = embed(pvec, [clip_words(train_c[c], 8000) for c in tr_ids])
    dv_pq = embed(pvec, [clip_words(r["text"], 8000) for r in dev_q])
    dv_pd = embed(pvec, [clip_words(dev_c[c], 8000) for c in dv_ids])
    te_pq = embed(pvec, [clip_words(r["text"], 8000) for r in test_q])
    te_pd = embed(pvec, [clip_words(test_c[c], 8000) for c in te_ids])
    tr_cent = gold_centroids(train_q, tr_ids, tr_pd)

    print("knn expansion grid", flush=True)
    best = {"f1": f1_8k, "name": "8k", "test": te_base, "dev": dv_base}
    for k in (3, 5, 8):
        for alpha in (0.15, 0.25, 0.4):
            dv_exp = knn_expand(dv_pq, tr_pq, tr_cent, k=k, alpha=alpha)
            dv_sc = dv_exp @ dv_pd.T
            if hasattr(dv_sc, "toarray"):
                dv_sc = dv_sc.toarray()
            f1, _, _ = show(f"dev/exp k={k} a={alpha}", dev_q, dv_ids, dv_sc)
            # blend with local 8k
            for mix in (0.15, 0.3, 0.5):
                fused = zscore(dv_base) + mix * zscore(np.asarray(dv_sc))
                f1b, _, _ = show(f"dev/8k+{mix}exp k={k} a={alpha}", dev_q, dv_ids, fused)
                if f1b > best["f1"] + 0.0008:
                    te_exp = knn_expand(te_pq, tr_pq, tr_cent, k=k, alpha=alpha)
                    te_sc = te_exp @ te_pd.T
                    if hasattr(te_sc, "toarray"):
                        te_sc = te_sc.toarray()
                    best = {
                        "f1": f1b, "name": f"8k+{mix}exp k={k} a={alpha}",
                        "dev": fused,
                        "test": zscore(te_base) + mix * zscore(np.asarray(te_sc)),
                    }
                    print(" KEEP", best["name"], f1b, flush=True)

    print("cite blend on current best", flush=True)
    dv_cite = cite_tfidf([r["text"] for r in dev_q], [dev_c[c] for c in dv_ids])
    te_cite = cite_tfidf([r["text"] for r in test_q], [test_c[c] for c in te_ids])
    for mix in (0.05, 0.08, 0.12):
        fused = zscore(best["dev"]) + mix * zscore(dv_cite)
        f1, _, _ = show(f"dev/best+{mix}cite", dev_q, dv_ids, fused)
        if f1 > best["f1"] + 0.0005:
            best = {"f1": f1, "name": best["name"] + f"+{mix}cite", "dev": fused,
                    "test": zscore(best["test"]) + mix * zscore(te_cite)}
            print(" KEEP", best["name"], f1, flush=True)

    print("sentence-max rerank of local 8k top20", flush=True)
    dv_smax = sent_max_for_top(dev_q, dev_c, dv_ids, dv_base, dv_vec, dv_qx, topn=20, max_sents=40)
    for mix in (0.15, 0.3, 0.5, 0.8):
        fused = rerank_top(dv_base, dv_smax, topn=20, mix=mix)
        f1, _, _ = show(f"dev/8k smax mix={mix}", dev_q, dv_ids, fused)
        if f1 > best["f1"] + 0.0008:
            te_smax = sent_max_for_top(test_q, test_c, te_ids, te_base, te_vec, te_qx, topn=20, max_sents=40)
            best = {"f1": f1, "name": f"8k-smax-{mix}", "dev": fused, "test": rerank_top(te_base, te_smax, topn=20, mix=mix)}
            print(" KEEP", best["name"], f1, flush=True)

    # smax on the expanded ranking too
    if best["name"] != "8k":
        for mix in (0.2, 0.4):
            fused = rerank_top(best["dev"], dv_smax, topn=20, mix=mix)
            f1, _, _ = show(f"dev/best smax mix={mix}", dev_q, dv_ids, fused)
            if f1 > best["f1"] + 0.0005:
                te_smax = sent_max_for_top(test_q, test_c, te_ids, te_base, te_vec, te_qx, topn=20, max_sents=40)
                best = {"f1": f1, "name": best["name"] + f"+smax{mix}", "dev": fused,
                        "test": rerank_top(best["test"], te_smax, topn=20, mix=mix)}
                print(" KEEP", best["name"], f1, flush=True)

    print("BEST", best["name"], best["f1"], "vs 8k", f1_8k, flush=True)
    with open(os.path.join(OUT, "choice.json"), "w") as handle:
        json.dump({"name": best["name"], "dev": best["f1"], "base": f1_8k}, handle, indent=2)
    ranked = rank_from_scores(test_q, te_ids, best["test"])
    write_zip(os.path.join(OUT, "submission.zip"), ranked)
    write_zip(os.path.join(SUB, "submission.zip"), ranked)
    with open(os.path.join(OUT, "prediction_dev.json"), "w") as handle:
        json.dump({qid: ids[:K] for qid, ids in rank_from_scores(dev_q, dv_ids, best["dev"]).items()}, handle)


if __name__ == "__main__":
    main()
