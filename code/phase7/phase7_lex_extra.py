"""One more lexical pass: longer 1-3 gram TF-IDF."""
import json
import os
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from phase7_lexboost import K, clip_words, load_split, micro_at_k, rank_from_scores, write_zip, zscore

OUT = r"C:\Users\Anshuman Raj\CS781\phase7_runs\lexboost"
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\lexboost"


def scores_for(queries, pool, ngram, cap, features, min_df=2):
    ids = list(pool)
    docs = [clip_words(pool[cid], cap, "ends") for cid in ids]
    qtexts = [clip_words(row["text"], cap, "ends") for row in queries]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngram), max_features=features, min_df=min_df, max_df=0.95,
        sublinear_tf=True, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b",
    )
    matrix = normalize(vectorizer.fit_transform(docs + qtexts))
    return ids, (matrix[len(docs):] @ matrix[:len(docs)].T).toarray()


def show(name, queries, ids, scores):
    ranked = rank_from_scores(queries, ids, scores)
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:24s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1, ranked, scores


def main():
    dev_q, dev_c = load_split("dev")
    configs = [
        ("w3_5k_300k", 3, 5000, 300000, 2),
        ("w3_7k_350k", 3, 7000, 350000, 2),
        ("w3_8k_400k", 3, 8000, 400000, 2),
        ("w3_5k_mindf1", 3, 5000, 300000, 1),
        ("w2_7k_250k", 2, 7000, 250000, 2),
        ("w4_5k_400k", 4, 5000, 400000, 2),
    ]
    best = {"f1": -1}
    stored = {}
    for name, ngram, cap, features, min_df in configs:
        print("scoring", name, flush=True)
        ids, scores = scores_for(dev_q, dev_c, ngram, cap, features, min_df)
        f1, ranked, scores = show(name, dev_q, ids, scores)
        stored[name] = scores
        if f1 > best["f1"]:
            best = {"f1": f1, "name": name, "ngram": ngram, "cap": cap, "features": features, "min_df": min_df, "scores": scores, "ids": ids}

    # blend top two stored
    names = sorted(stored, key=lambda n: show(n + "/re", dev_q, best["ids"], stored[n])[0], reverse=True)
    print("best single", best["name"], best["f1"], flush=True)
    a, b = names[0], names[1]
    mix_best = best
    for coef in (0.1, 0.2, 0.35, 0.5):
        mix = zscore(stored[a]) + coef * zscore(stored[b])
        f1, _, _ = show(f"{a}+{coef}{b}", dev_q, best["ids"], mix)
        if f1 > mix_best["f1"]:
            mix_best = {"f1": f1, "name": f"{a}+{coef}{b}", "a": a, "b": b, "coef": coef, "scores": mix}

    print("WINNER", mix_best["name"], mix_best["f1"], flush=True)
    test_q, test_c = load_split("test")
    if "a" in mix_best:
        cfg = {n: next(c for c in configs if c[0] == n) for n in (mix_best["a"], mix_best["b"])}
        t_ids, sa = scores_for(test_q, test_c, *cfg[mix_best["a"]][1:])
        _, sb = scores_for(test_q, test_c, *cfg[mix_best["b"]][1:])
        test_scores = zscore(sa) + mix_best["coef"] * zscore(sb)
    else:
        t_ids, test_scores = scores_for(test_q, test_c, best["ngram"], best["cap"], best["features"], best["min_df"])
        mix_best = best
    ranked = rank_from_scores(test_q, t_ids, test_scores)
    write_zip(os.path.join(OUT, "submission.zip"), ranked)
    write_zip(os.path.join(SUB, "submission.zip"), ranked)
    with open(os.path.join(OUT, "best.json"), "w") as handle:
        json.dump({k: v for k, v in mix_best.items() if k != "scores"}, handle, indent=2)
    print("wrote", mix_best["name"], mix_best["f1"], flush=True)


if __name__ == "__main__":
    main()
