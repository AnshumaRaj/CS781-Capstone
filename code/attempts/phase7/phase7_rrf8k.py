"""RRF the 8k TF-IDF winner with completed neural lists. CPU only."""
from __future__ import annotations

import json
import os
from collections import defaultdict

from phase7_lexboost import K, load_split, micro_at_k, rank_from_scores, write_zip
from phase7_lex_extra import scores_for, show

KAGGLE = r"C:\Users\Anshuman Raj\CS781\phase7_kaggle"
OUT = r"C:\Users\Anshuman Raj\CS781\phase7_runs\rrf8k"
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\submissions\rrf8k"
NEURAL = {
    "eff_lorma": os.path.join(KAGGLE, "p7-eff-lorma", "eff_lorma"),
    "eff_partial": os.path.join(KAGGLE, "p7-eff-partial", "eff_partial"),
    "dec_lorma": os.path.join(KAGGLE, "p7-dec-lorma", "dec_lorma"),
    "hyb_adapter": os.path.join(KAGGLE, "p7-hyb-adapter", "hyb_adapter"),
    "tfidf3k": os.path.join(KAGGLE, "p7-tfidf", "tfidf"),
}


def load_ranked(folder, name):
    path = os.path.join(folder, name)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def rrf(queries, lists, weights, k=60):
    out = {}
    for row in queries:
        qid = row["id"]
        scores = defaultdict(float)
        for ranked, weight in zip(lists, weights):
            for rank, cid in enumerate(ranked[qid]):
                if cid == qid:
                    continue
                scores[cid] += weight / (k + rank + 1)
        out[qid] = [cid for cid, _ in sorted(scores.items(), key=lambda item: -item[1])][:50]
    return out


def eval_ranked(name, queries, ranked):
    f1, precision, recall, tp, fp, fn = micro_at_k(queries, ranked)
    print(f"{name:40s} F1={f1:.4f} P={precision:.4f} R={recall:.4f} tp={tp}", flush=True)
    return f1


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB, exist_ok=True)
    print("8k tfidf", flush=True)
    dev_q, dev_c = load_split("dev")
    ids, scores8 = scores_for(dev_q, dev_c, 3, 8000, 400000)
    f1_8k, ranked8, _ = show("lex8k", dev_q, ids, scores8)
    ranked8 = {row["id"]: ranked8[row["id"]] for row in dev_q}

    others = {}
    for name, folder in NEURAL.items():
        others[name] = load_ranked(folder, "ranked_dev.json")
        eval_ranked(name, dev_q, others[name])

    best = {"f1": f1_8k, "spec": ("lex8k",), "w": (1.0,), "ranked": ranked8}
    trials = [
        ("8k+eff_lorma", [ranked8, others["eff_lorma"]]),
        ("8k+eff_partial", [ranked8, others["eff_partial"]]),
        ("8k+both_eff", [ranked8, others["eff_lorma"], others["eff_partial"]]),
        ("8k+3k", [ranked8, others["tfidf3k"]]),
        ("8k+3k+eff_lorma", [ranked8, others["tfidf3k"], others["eff_lorma"]]),
        ("8k+all_neural", [ranked8, others["eff_lorma"], others["eff_partial"], others["dec_lorma"], others["hyb_adapter"]]),
    ]
    for label, lists in trials:
        n = len(lists)
        if n == 2:
            grid = [(1.0, w) for w in (0.05, 0.1, 0.15, 0.25, 0.4, 0.6)]
        elif n == 3:
            grid = [(1.0, 0.15, 0.1), (1.0, 0.2, 0.15), (1.0, 0.1, 0.2), (1.0, 0.25, 0.05)]
        else:
            grid = [(1.0, 0.08, 0.08, 0.04, 0.04), (1.0, 0.12, 0.12, 0.05, 0.05)]
        for weights in grid:
            ranked = rrf(dev_q, lists, weights)
            f1 = eval_ranked(f"{label} {weights}", dev_q, ranked)
            if f1 > best["f1"] + 0.0005:
                best = {"f1": f1, "spec": (label, weights), "ranked": ranked}
                print(" keep", best["spec"], f1, flush=True)

    print("BEST", best["f1"], best["spec"], flush=True)
    with open(os.path.join(OUT, "blend.json"), "w") as handle:
        json.dump({"f1": best["f1"], "spec": [str(x) for x in best["spec"]]}, handle, indent=2)

    print("test 8k", flush=True)
    test_q, test_c = load_split("test")
    t_ids, t_scores = scores_for(test_q, test_c, 3, 8000, 400000)
    ranked8_test = rank_from_scores(test_q, t_ids, t_scores)
    test_others = {}
    for name, folder in NEURAL.items():
        test_others[name] = load_ranked(folder, "prediction.json")

    spec = best["spec"]
    if spec[0] == "lex8k" or best["f1"] <= f1_8k + 0.0005:
        ranked_test = ranked8_test
        print("using pure 8k", flush=True)
    else:
        label, weights = spec
        mapping = {
            "8k+eff_lorma": [ranked8_test, test_others["eff_lorma"]],
            "8k+eff_partial": [ranked8_test, test_others["eff_partial"]],
            "8k+both_eff": [ranked8_test, test_others["eff_lorma"], test_others["eff_partial"]],
            "8k+3k": [ranked8_test, test_others["tfidf3k"]],
            "8k+3k+eff_lorma": [ranked8_test, test_others["tfidf3k"], test_others["eff_lorma"]],
            "8k+all_neural": [ranked8_test, test_others["eff_lorma"], test_others["eff_partial"],
                              test_others["dec_lorma"], test_others["hyb_adapter"]],
        }
        ranked_test = rrf(test_q, mapping[label], weights)
        print("using", label, weights, flush=True)

    write_zip(os.path.join(OUT, "submission.zip"), ranked_test)
    write_zip(os.path.join(SUB, "submission.zip"), ranked_test)
    with open(os.path.join(OUT, "prediction_dev.json"), "w") as handle:
        json.dump({qid: ids[:K] for qid, ids in best["ranked"].items()}, handle)


if __name__ == "__main__":
    main()
