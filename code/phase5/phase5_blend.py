"""Weighted logit blend: base run + w * other run, scored with the honest 2-fold per-label threshold estimate.

    python phase5_blend.py BASE_RUN OTHER_RUN [--write OUT_DIR --weight W]
"""
import argparse
import json
import os
import zipfile

import numpy as np

from phase5_ensemble import find_runs
from phase5_lsi import DEFAULT_DATA, apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report
from phase5_stack import load_logits


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def standardize(dev, test):
    mean, std = dev.mean(), dev.std() + 1e-6
    return (dev - mean) / std, (test - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("others", nargs="+")
    parser.add_argument("--weight", type=float, default=-1.0)
    parser.add_argument("--write", default="")
    args = parser.parse_args()

    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)

    base_dev, base_test = standardize(*load_logits(find_runs([args.base])[0], dev_ids, test_ids))
    others = [standardize(*load_logits(folder, dev_ids, test_ids)) for folder in find_runs(args.others)]
    other_dev = np.mean([item[0] for item in others], axis=0)
    other_test = np.mean([item[1] for item in others], axis=0)

    weights = [args.weight] if args.weight >= 0 else [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    best = (-1.0, 0.0)
    for weight in weights:
        value = threshold_report(gold, sigmoid(base_dev + weight * other_dev))["f1_per_label_crossval"]
        print(f"w {weight:<5} crossval {value:.4f}", flush=True)
        best = max(best, (value, weight))
    weight = best[1]

    if args.write:
        dev_probs = sigmoid(base_dev + weight * other_dev)
        test_probs = sigmoid(base_test + weight * other_test)
        report = threshold_report(gold, dev_probs)
        cuts = per_label_cuts(gold, dev_probs)
        report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
        report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
        report["weight"] = weight
        report["members"] = [args.base] + args.others
        os.makedirs(args.write, exist_ok=True)
        test_pred = apply_cuts(test_probs, cuts)
        predictions = [
            {"id": item, "pred": [int(label) for label in np.flatnonzero(test_pred[index])]}
            for index, item in enumerate(test_ids)
        ]
        pred_path = os.path.join(args.write, "prediction.json")
        with open(pred_path, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle)
        with zipfile.ZipFile(os.path.join(args.write, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(pred_path, arcname="prediction.json")
        with open(os.path.join(args.write, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("wrote", os.path.join(args.write, "submission.zip"), json.dumps(report["f1_per_label_crossval"]), flush=True)


if __name__ == "__main__":
    main()
