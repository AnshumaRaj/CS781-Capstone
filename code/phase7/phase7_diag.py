"""Where does gold sit in the 8k ranking? CPU only."""
import numpy as np
from phase7_lexboost import load_split, clip_words, rank_from_scores
from phase7_rerank import vec_fit

dev_q, dev_c = load_split("dev")
ids = list(dev_c)
docs = [dev_c[c] for c in ids]
qtexts = [r["text"] for r in dev_q]
scores = vec_fit(qtexts, docs)
id_to_j = {c: i for i, c in enumerate(ids)}
cuts = [7, 20, 50, 100, 200]
hits = {k: 0 for k in cuts}
total = 0
ranks = []
missing_pool = 0
for i, row in enumerate(dev_q):
    order = np.argsort(-scores[i])
    rank_of = {ids[j]: r for r, j in enumerate(order) if ids[j] != row["id"]}
    for g in row["rel"]:
        total += 1
        if g not in id_to_j:
            missing_pool += 1
            continue
        r = rank_of.get(g, 10**9)
        ranks.append(r)
        for k in cuts:
            if r < k:
                hits[k] += 1
ranks = np.array(ranks)
print("gold", total, "missing_pool", missing_pool)
for k in cuts:
    print(f"recall@{k}", round(hits[k] / total, 4), "tp", hits[k])
print("gold rank p50", int(np.median(ranks)), "p75", int(np.percentile(ranks, 75)),
      "p90", int(np.percentile(ranks, 90)), "mean", round(float(ranks.mean()), 1))
# queries with 0 hits in top7
zero = sum(1 for i, row in enumerate(dev_q) if not set(row["rel"]) & set(rank_from_scores([row], ids, scores[i:i+1])[row["id"]][:7]))
print("queries with 0/7", zero, "/", len(dev_q))
