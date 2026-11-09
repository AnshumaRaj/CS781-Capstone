"""Stronger lexical PCR: longer TF-IDF, PRF, fielded head, window max-sim."""
from __future__ import annotations

import json
import os
import re
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from phase7_lexboost import (
    K, OUT, SUB, clip_words, load_split, micro_at_k, rank_from_scores, write_zip, zscore,
)

FIELD = re.compile(
    r"(?:CITATION|HEADNOTE|ACT|JUDGMENT|PETITIONER|RESPONDENT|BENCH)\s*:",
    re.I,
)


def show(name, queries, ids, scores):
    ranked = rank_from_scores(queries, ids, scores)
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:32s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1, ranked, scores


def header_text(text, cap=1500):
    match = re.search(r"HEADNOTE\s*:", text, re.I)
    if match:
        return clip_words(text[: match.start()] + " " + text[match.start(): match.start() + 4000], cap, "head")
    return clip_words(text, cap, "head")


def vectorize(qtexts, docs, ngram=3, features=400000, min_df=2, binary=False):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram),
        max_features=features,
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=not binary,
        lowercase=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b",
        binary=binary,
    )
    matrix = normalize(vectorizer.fit_transform(list(docs) + list(qtexts)))
    doc_x = matrix[:len(docs)]
    query_x = matrix[len(docs):]
    return query_x, doc_x, (query_x @ doc_x.T).toarray()


def prf(query_x, doc_x, scores, cand_ids, query_ids, n=3, alpha=0.7):
    out = np.zeros_like(scores)
    for i, qid in enumerate(query_ids):
        order = np.argsort(-scores[i])
        picked = []
        for index in order:
            if cand_ids[index] == qid:
                continue
            picked.append(index)
            if len(picked) >= n:
                break
        mean = np.asarray(doc_x[picked].mean(axis=0))
        query = query_x[i].toarray()
        expanded = normalize(alpha * query + (1.0 - alpha) * mean)
        out[i] = np.asarray(expanded @ doc_x.T).ravel()
    return out


def window_docs(texts, width=2500, count=5):
    out = []
    owners = []
    for index, text in enumerate(texts):
        words = text.split()
        if len(words) <= width:
            chunks = [text]
        else:
            starts = [int(round(i * max(len(words) - width, 0) / max(count - 1, 1))) for i in range(count)]
            chunks = [" ".join(words[s:s + width]) for s in starts]
        for chunk in chunks:
            out.append(chunk)
            owners.append(index)
    return out, owners


def window_max_sim(qtexts, docs, ngram=3, features=300000, width=2500, count=5):
    chunks, owners = window_docs(docs, width, count)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram), max_features=features, min_df=2, max_df=0.95,
        sublinear_tf=True, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b",
    )
    matrix = normalize(vectorizer.fit_transform(chunks + list(qtexts)))
    chunk_x = matrix[:len(chunks)]
    query_x = matrix[len(chunks):]
    sim = (query_x @ chunk_x.T).toarray()
    n_docs = len(docs)
    out = np.full((len(qtexts), n_docs), -1.0, dtype=np.float32)
    owners = np.asarray(owners)
    for d in range(n_docs):
        cols = np.where(owners == d)[0]
        out[:, d] = sim[:, cols].max(axis=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    print("load dev", flush=True)
    dev_q, dev_c = load_split("dev")
    ids = list(dev_c)
    raw_docs = [dev_c[cid] for cid in ids]
    raw_q = [row["text"] for row in dev_q]
    qids = [row["id"] for row in dev_q]

    variants = {}
    plans = [
        ("w3_8k", [clip_words(t, 8000) for t in raw_q], [clip_words(t, 8000) for t in raw_docs], dict(ngram=3, features=400000)),
        ("w3_12k", [clip_words(t, 12000) for t in raw_q], [clip_words(t, 12000) for t in raw_docs], dict(ngram=3, features=450000)),
        ("w3_full", raw_q, raw_docs, dict(ngram=3, features=500000)),
        ("w3_8k_bin", [clip_words(t, 8000) for t in raw_q], [clip_words(t, 8000) for t in raw_docs], dict(ngram=3, features=400000, binary=True)),
        ("head_w3", [header_text(t) for t in raw_q], [header_text(t) for t in raw_docs], dict(ngram=3, features=200000)),
    ]
    stored_xd = {}
    for name, qtexts, docs, kwargs in plans:
        print("vec", name, flush=True)
        query_x, doc_x, scores = vectorize(qtexts, docs, **kwargs)
        f1, _, scores = show(name, dev_q, ids, scores)
        variants[name] = scores
        stored_xd[name] = (query_x, doc_x, qtexts, docs, kwargs)

    print("window max-sim", flush=True)
    win = window_max_sim(raw_q, raw_docs)
    f1, _, win = show("window_max", dev_q, ids, win)
    variants["window_max"] = win

    print("PRF", flush=True)
    base_name = max(("w3_8k", "w3_12k", "w3_full"), key=lambda n: show(n + "/chk", dev_q, ids, variants[n])[0])
    query_x, doc_x, _, _, _ = stored_xd[base_name]
    best_prf = {"f1": -1}
    for n in (2, 3, 5, 8):
        for alpha in (0.5, 0.65, 0.8, 0.9):
            scores = prf(query_x, doc_x, variants[base_name], ids, qids, n=n, alpha=alpha)
            f1, _, scores = show(f"prf n={n} a={alpha} {base_name}", dev_q, ids, scores)
            variants[f"prf_{n}_{alpha}"] = scores
            if f1 > best_prf["f1"]:
                best_prf = {"f1": f1, "n": n, "alpha": alpha, "name": f"prf_{n}_{alpha}"}

    print("blend", flush=True)
    ranked_names = sorted(variants, key=lambda n: show(n + "/b", dev_q, ids, variants[n])[0], reverse=True)
    current = zscore(variants[ranked_names[0]])
    weights = {ranked_names[0]: 1.0}
    best_f1 = show("start " + ranked_names[0], dev_q, ids, current)[0]
    best_scores = current.copy()
    for name in ranked_names[1:8]:
        trial_best = None
        for coef in (0.1, 0.2, 0.35, 0.5, 0.75):
            trial = current + coef * zscore(variants[name])
            f1 = show(f"+{coef} {name}", dev_q, ids, trial)[0]
            if f1 > (trial_best[0] if trial_best else -1):
                trial_best = (f1, coef, trial)
        if trial_best and trial_best[0] > best_f1 + 0.0008:
            current = trial_best[2]
            weights[name] = trial_best[1]
            best_f1 = trial_best[0]
            best_scores = current.copy()
            print("keep", name, trial_best[1], best_f1, flush=True)
        else:
            print("skip", name, flush=True)

    print("BEST DEV", best_f1, weights, "prf", best_prf, flush=True)
    with open(os.path.join(OUT, "v3.json"), "w", encoding="utf-8") as handle:
        json.dump({"f1": best_f1, "weights": weights, "prf": best_prf,
                   "singles": {n: float(show(n + "/s", dev_q, ids, variants[n])[0]) for n in ranked_names[:12]}}, handle, indent=2)

    print("test", flush=True)
    test_q, test_c = load_split("test")
    t_ids = list(test_c)
    t_docs_raw = [test_c[cid] for cid in t_ids]
    t_q_raw = [row["text"] for row in test_q]
    t_qids = [row["id"] for row in test_q]

    def make_variant(name):
        if name == "w3_8k":
            return vectorize([clip_words(t, 8000) for t in t_q_raw], [clip_words(t, 8000) for t in t_docs_raw], ngram=3, features=400000)
        if name == "w3_12k":
            return vectorize([clip_words(t, 12000) for t in t_q_raw], [clip_words(t, 12000) for t in t_docs_raw], ngram=3, features=450000)
        if name == "w3_full":
            return vectorize(t_q_raw, t_docs_raw, ngram=3, features=500000)
        if name == "w3_8k_bin":
            return vectorize([clip_words(t, 8000) for t in t_q_raw], [clip_words(t, 8000) for t in t_docs_raw], ngram=3, features=400000, binary=True)
        if name == "head_w3":
            return vectorize([header_text(t) for t in t_q_raw], [header_text(t) for t in t_docs_raw], ngram=3, features=200000)
        return None

    test_parts = {}
    # always compute needed ingredients
    needed = set(weights)
    if any(n.startswith("prf_") for n in needed):
        needed.add(base_name)
    needed |= {base_name, "w3_8k", "w3_12k", "w3_full"}
    for name in ["w3_8k", "w3_12k", "w3_full", "w3_8k_bin", "head_w3"]:
        if name in needed or name in weights:
            print("test vec", name, flush=True)
            qx, dx, sc = make_variant(name)
            test_parts[name] = sc
            if name == base_name:
                test_prf_q, test_prf_d = qx, dx
    if "window_max" in weights:
        print("test window", flush=True)
        test_parts["window_max"] = window_max_sim(t_q_raw, t_docs_raw)
    for name in list(weights):
        if name.startswith("prf_"):
            _, n_s, a_s = name.split("_", 2)
            n, alpha = int(n_s), float(a_s)
            print("test prf", n, alpha, flush=True)
            test_parts[name] = prf(test_prf_q, test_prf_d, test_parts[base_name], t_ids, t_qids, n=n, alpha=alpha)

    test_scores = None
    for name, coef in weights.items():
        part = zscore(test_parts[name])
        test_scores = part * coef if test_scores is None else test_scores + part * coef
    ranked = rank_from_scores(test_q, t_ids, test_scores)
    dest = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\lexv3"
    os.makedirs(dest, exist_ok=True)
    write_zip(os.path.join(OUT, "submission_v3.zip"), ranked)
    write_zip(os.path.join(dest, "submission.zip"), ranked)
    with open(os.path.join(OUT, "prediction_dev_v3.json"), "w", encoding="utf-8") as handle:
        json.dump({qid: ids_[:K] for qid, ids_ in rank_from_scores(dev_q, ids, best_scores).items()}, handle)
    print("done", best_f1, flush=True)


if __name__ == "__main__":
    main()
