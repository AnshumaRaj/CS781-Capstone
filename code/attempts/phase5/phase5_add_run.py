"""Add one more run to the current best efficient blend; score over five dev splits.

    python phase5_add_run.py RUN_DIR [--write OUT_DIR]
"""
import argparse
import json
import os
import zipfile

import numpy as np

from phase5_blend import standardize
from phase5_fineblend import PATHS
from phase5_lsi import apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report
from phase5_stack import load_logits


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def mix(rows, weight):
    return (
        rows["tfidf"]
        + 0.45 * 0.5 * (rows["part4"] + rows["ctx"])
        + 0.1 * (rows["long"] + rows["s2"] + rows["w384"]) / 3
        + weight * rows["extra"]
    )


def robust(gold, probs):
    return float(np.mean([threshold_report(gold, probs, seed=seed)["f1_per_label_crossval"] for seed in range(5)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--write", default="")
    args = parser.parse_args()

    root = os.path.join(r"C:\Users\Anshuman Raj\CS781\phase5data", "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)
    paths = dict(PATHS)
    paths["extra"] = args.run
    dev, test = {}, {}
    for name, path in paths.items():
        dev[name], test[name] = standardize(*load_logits(path, dev_ids, test_ids))

    best = (0.0, 0.0)
    for weight in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        value = robust(gold, sigmoid(mix(dev, weight)))
        print(f"extra {weight} {value:.4f}", flush=True)
        best = max(best, (value, weight))
    print("best", round(best[0], 4), best[1], flush=True)

    if args.write and best[1] > 0:
        dev_probs, test_probs = sigmoid(mix(dev, best[1])), sigmoid(mix(test, best[1]))
        cuts = per_label_cuts(gold, dev_probs)
        os.makedirs(args.write, exist_ok=True)
        pred = apply_cuts(test_probs, cuts)
        predictions = [
            {"id": item, "pred": [int(label) for label in np.flatnonzero(pred[index])]}
            for index, item in enumerate(test_ids)
        ]
        pred_path = os.path.join(args.write, "prediction.json")
        with open(pred_path, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle)
        with zipfile.ZipFile(os.path.join(args.write, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(pred_path, arcname="prediction.json")
        report = threshold_report(gold, dev_probs)
        report["robust_crossval_5_splits"] = best[0]
        report["extra_run"] = args.run
        report["extra_weight"] = best[1]
        report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
        report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
        with open(os.path.join(args.write, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("wrote", os.path.join(args.write, "submission.zip"), flush=True)


if __name__ == "__main__":
    main()
