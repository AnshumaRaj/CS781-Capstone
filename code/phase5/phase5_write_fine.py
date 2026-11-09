"""Write the best fine blend: TF-IDF + 0.45 * mean(part4, ctx) + 0.1 * mean(long, s2, w384)."""
import json
import os
import zipfile

import numpy as np

from phase5_blend import standardize
from phase5_fineblend import PATHS
from phase5_lsi import apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report
from phase5_stack import load_logits

OUT = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p5_sub\efficient_fineblend_tfidf"


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def main():
    root = os.path.join(r"C:\Users\Anshuman Raj\CS781\phase5data", "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)
    dev, test = {}, {}
    for name, path in PATHS.items():
        dev[name], test[name] = standardize(*load_logits(path, dev_ids, test_ids))
    dev_logits = dev["tfidf"] + 0.45 * 0.5 * (dev["part4"] + dev["ctx"]) + 0.1 * (dev["long"] + dev["s2"] + dev["w384"]) / 3
    test_logits = test["tfidf"] + 0.45 * 0.5 * (test["part4"] + test["ctx"]) + 0.1 * (test["long"] + test["s2"] + test["w384"]) / 3
    dev_probs, test_probs = sigmoid(dev_logits), sigmoid(test_logits)
    report = threshold_report(gold, dev_probs)
    cuts = per_label_cuts(gold, dev_probs)
    report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
    report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
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
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("crossval", round(report["f1_per_label_crossval"], 4), flush=True)


if __name__ == "__main__":
    main()
