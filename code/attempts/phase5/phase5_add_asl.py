"""Add the laptop asymmetric-loss run to the current best blend and keep it only if the score rises."""
import json
import os
import zipfile

import numpy as np

from phase5_blend import standardize
from phase5_fineblend import PATHS
from phase5_lsi import apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report
from phase5_stack import load_logits

ASL = r"C:\Users\Anshuman Raj\CS781\phase5_runs\eff_asl_laptop"
OUT = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p5_sub\efficient_fineblend_asl"


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def mix(rows, weight):
    return (
        rows["tfidf"]
        + 0.45 * 0.5 * (rows["part4"] + rows["ctx"])
        + 0.1 * (rows["long"] + rows["s2"] + rows["w384"]) / 3
        + weight * rows["asl"]
    )


def main():
    root = os.path.join(r"C:\Users\Anshuman Raj\CS781\phase5data", "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)
    dev, test = {}, {}
    paths = dict(PATHS)
    paths["asl"] = ASL
    for name, path in paths.items():
        dev[name], test[name] = standardize(*load_logits(path, dev_ids, test_ids))
    best = (0.0, 0.0)
    for weight in (0.0, 0.05, 0.1, 0.15, 0.25, 0.4):
        value = threshold_report(gold, sigmoid(mix(dev, weight)))["f1_per_label_crossval"]
        print(f"asl {weight} {value:.4f}", flush=True)
        best = max(best, (value, weight))
    print("best", best, flush=True)
    if best[1] == 0.0 or best[0] < 0.4225:
        return
    dev_probs = sigmoid(mix(dev, best[1]))
    test_probs = sigmoid(mix(test, best[1]))
    cuts = per_label_cuts(gold, dev_probs)
    os.makedirs(OUT, exist_ok=True)
    pred = apply_cuts(test_probs, cuts)
    predictions = [
        {"id": item, "pred": [int(label) for label in np.flatnonzero(pred[index])]}
        for index, item in enumerate(test_ids)
    ]
    pred_path = os.path.join(OUT, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    with zipfile.ZipFile(os.path.join(OUT, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    report = threshold_report(gold, dev_probs)
    report["weight"] = best[1]
    report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
    report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", round(report["f1_per_label_crossval"], 4), flush=True)


if __name__ == "__main__":
    main()
