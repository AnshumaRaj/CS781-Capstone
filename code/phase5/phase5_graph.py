"""Boost the current best blend with label co-occurrence from train.

    python phase5_graph.py [--write OUT_DIR]
"""
import json
import os
import zipfile

import numpy as np

from phase5_blend import standardize
from phase5_fineblend import PATHS
from phase5_lsi import DEFAULT_DATA, apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report
from phase5_stack import load_logits


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def blend_logits(dev, test):
    def mix(rows):
        return (
            rows["tfidf"]
            + 0.45 * 0.5 * (rows["part4"] + rows["ctx"])
            + 0.1 * (rows["long"] + rows["s2"] + rows["w384"]) / 3
        )
    return mix(dev), mix(test)


def cond_prob(gold):
    present = gold.sum(0).clip(min=1)
    joint = gold.T @ gold
    cond = joint / present[:, None]
    np.fill_diagonal(cond, 0.0)
    return cond


def boost(logits, cond, alpha):
    probs = sigmoid(logits)
    extra = probs @ cond
    return logits + alpha * extra


def main():
    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    train = read_rows(os.path.join(root, "Train", "train.json"), labeled=True)
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)
    cond = cond_prob(label_matrix(train))
    scaled = {name: standardize(*load_logits(path, dev_ids, test_ids)) for name, path in PATHS.items()}
    dev = {name: pair[0] for name, pair in scaled.items()}
    test = {name: pair[1] for name, pair in scaled.items()}
    base_dev, base_test = blend_logits(dev, test)
    best = (0.0, 0.0)
    for alpha in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        value = threshold_report(gold, sigmoid(boost(base_dev, cond, alpha)))["f1_per_label_crossval"]
        print(f"alpha {alpha} {value:.4f}", flush=True)
        best = max(best, (value, alpha))
    print("best", best, flush=True)
    if best[1] == 0.0 or best[0] < 0.423:
        return
    out = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p5_sub\efficient_graph_tfidf"
    os.makedirs(out, exist_ok=True)
    dev_probs = sigmoid(boost(base_dev, cond, best[1]))
    test_probs = sigmoid(boost(base_test, cond, best[1]))
    cuts = per_label_cuts(gold, dev_probs)
    pred = apply_cuts(test_probs, cuts)
    predictions = [
        {"id": item, "pred": [int(label) for label in np.flatnonzero(pred[index])]}
        for index, item in enumerate(test_ids)
    ]
    pred_path = os.path.join(out, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    report = threshold_report(gold, dev_probs)
    report["alpha"] = best[1]
    report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
    report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", os.path.join(out, "submission.zip"), flush=True)


if __name__ == "__main__":
    main()
