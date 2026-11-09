"""Fast DEV fielded RRF around the 8k winner. No train 4320 fit."""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from phase7_lexboost import K, clip_words, load_split, micro_at_k, rank_from_scores, write_zip, zscore

OUT = r"C:\Users\Anshuman Raj\CS781\phase7_runs\fields"
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\fields"


def sents_of(value):
    if isinstance(value, dict):
        t = value.get("text") or []
        if isinstance(t, list):
            return [str(s) for s in t if str(s).strip()]
        return str(t).split(". ")
    if isinstance(value, list):
        return [str(s) for s in value]
    return str(value).split(". ")


def tfidf(qtexts, docs, ngram=3, features=400000, min_df=2):
    vec = TfidfVectorizer(
        ngram_range=(1, ngram), max_features=features, min_df=min_df, max_df=0.95,
        sublinear_tf=True, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b",
    )
    matrix = normalize(vec.fit_transform(list(docs) + list(qtexts)))
    return (matrix[len(docs):] @ matrix[:len(docs)].T).toarray()


def show(name, queries, ids, scores):
    ranked = rank_from_scores(queries, ids, scores)
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:36s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1, ranked, scores


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    with open(r"C:\Users\Anshuman Raj\CS781\phase7data\Phase_7_PCR\Dev\dev.json", encoding="utf-8") as h:
        raw_q = json.load(h)
    with open(r"C:\Users\Anshuman Raj\CS781\phase7data\Phase_7_PCR\Dev\dev_candidates.json", encoding="utf-8") as h:
        raw_c = json.load(h)
    dev_q, dev_c = load_split("dev")
    ids = list(dev_c)
    q_sents = [sents_of(raw_q[r["id"]]) for r in dev_q]
    d_sents = [sents_of(raw_c[c]) for c in ids]

    variants = {}
    plans = {
        "8k": ([clip_words(r["text"], 8000) for r in dev_q], [clip_words(dev_c[c], 8000) for c in ids], 3, 400000),
        "full_w2": ([r["text"] for r in dev_q], [dev_c[c] for c in ids], 2, 400000),
        "full_w3": ([r["text"] for r in dev_q], [dev_c[c] for c in ids], 3, 400000),
        "tail2k": ([clip_words(r["text"], 2000, "tail") for r in dev_q], [clip_words(dev_c[c], 2000, "tail") for c in ids], 3, 200000),
        "head2k": ([clip_words(r["text"], 2000, "head") for r in dev_q], [clip_words(dev_c[c], 2000, "head") for c in ids], 3, 200000),
        "facts": ([" ".join(s[:30]) for s in q_sents], [" ".join(s[:30]) for s in d_sents], 3, 250000),
        "hold": ([" ".join(s[-12:]) for s in q_sents], [" ".join(s[-12:]) for s in d_sents], 3, 200000),
        "mid": ([" ".join(s[10:40]) for s in q_sents], [" ".join(s[10:40]) for s in d_sents], 3, 200000),
        "q8k_dfull": ([clip_words(r["text"], 8000) for r in dev_q], [dev_c[c] for c in ids], 3, 400000),
    }
    for name, (qs, ds, ng, feat) in plans.items():
        print("score", name, flush=True)
        sc = tfidf(qs, ds, ngram=ng, features=feat)
        f1, _, sc = show(name, dev_q, ids, sc)
        variants[name] = sc

    # greedy zscore blend starting at 8k
    current = zscore(variants["8k"])
    weights = {"8k": 1.0}
    best_f1 = show("start 8k", dev_q, ids, current)[0]
    best_sc = current.copy()
    order = sorted(variants, key=lambda n: show(n + "/ord", dev_q, ids, variants[n])[0], reverse=True)
    for name in order:
        if name == "8k":
            continue
        trial_best = None
        for coef in (0.08, 0.12, 0.2, 0.35, 0.5):
            trial = current + coef * zscore(variants[name])
            f1 = show(f"+{coef} {name}", dev_q, ids, trial)[0]
            if f1 > (trial_best[0] if trial_best else -1):
                trial_best = (f1, coef, trial)
        if trial_best and trial_best[0] > best_f1 + 0.0008:
            current = trial_best[2]
            weights[name] = trial_best[1]
            best_f1 = trial_best[0]
            best_sc = current.copy()
            print("KEEP", name, trial_best[1], best_f1, flush=True)
        else:
            print("skip", name, flush=True)

    print("BEST DEV", best_f1, weights, flush=True)
    with open(os.path.join(OUT, "choice.json"), "w") as h:
        json.dump({"f1": best_f1, "weights": weights}, h, indent=2)

    if best_f1 <= 0.4302 + 0.0005 and set(weights) == {"8k"}:
        print("no gain, not packing a new primary", flush=True)
        return

    print("test", flush=True)
    test_q, test_c = load_split("test")
    with open(r"C:\Users\Anshuman Raj\CS781\phase7data\Phase_7_PCR\Test\test.json", encoding="utf-8") as h:
        raw_tq = json.load(h)
    with open(r"C:\Users\Anshuman Raj\CS781\phase7data\Phase_7_PCR\Test\test_candidates.json", encoding="utf-8") as h:
        raw_tc = json.load(h)
    t_ids = list(test_c)
    tq_sents = [sents_of(raw_tq[r["id"]]) for r in test_q]
    td_sents = [sents_of(raw_tc[c]) for c in t_ids]
    makers = {
        "8k": lambda: tfidf([clip_words(r["text"], 8000) for r in test_q], [clip_words(test_c[c], 8000) for c in t_ids], 3, 400000),
        "full_w2": lambda: tfidf([r["text"] for r in test_q], [test_c[c] for c in t_ids], 2, 400000),
        "full_w3": lambda: tfidf([r["text"] for r in test_q], [test_c[c] for c in t_ids], 3, 400000),
        "tail2k": lambda: tfidf([clip_words(r["text"], 2000, "tail") for r in test_q], [clip_words(test_c[c], 2000, "tail") for c in t_ids], 3, 200000),
        "head2k": lambda: tfidf([clip_words(r["text"], 2000, "head") for r in test_q], [clip_words(test_c[c], 2000, "head") for c in t_ids], 3, 200000),
        "facts": lambda: tfidf([" ".join(s[:30]) for s in tq_sents], [" ".join(s[:30]) for s in td_sents], 3, 250000),
        "hold": lambda: tfidf([" ".join(s[-12:]) for s in tq_sents], [" ".join(s[-12:]) for s in td_sents], 3, 200000),
        "mid": lambda: tfidf([" ".join(s[10:40]) for s in tq_sents], [" ".join(s[10:40]) for s in td_sents], 3, 200000),
        "q8k_dfull": lambda: tfidf([clip_words(r["text"], 8000) for r in test_q], [test_c[c] for c in t_ids], 3, 400000),
    }
    test_sc = None
    for name, coef in weights.items():
        print("test", name, coef, flush=True)
        part = zscore(makers[name]())
        test_sc = part * coef if test_sc is None else test_sc + part * coef
    write_zip(os.path.join(OUT, "submission.zip"), rank_from_scores(test_q, t_ids, test_sc))
    write_zip(os.path.join(SUB, "submission.zip"), rank_from_scores(test_q, t_ids, test_sc))
    with open(os.path.join(OUT, "prediction_dev.json"), "w") as h:
        json.dump({qid: v[:K] for qid, v in rank_from_scores(dev_q, ids, best_sc).items()}, h)
    print("packed", best_f1, flush=True)


if __name__ == "__main__":
    main()
