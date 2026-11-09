"""Last-day efficient blend: TF-IDF + the two train+dev neural runs.

Dev scores here are leaked (those models saw labeled dev). The zip is for the hidden test set.
"""
import json
import os
import zipfile

import numpy as np

from phase5_blend import standardize
from phase5_lsi import apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report
from phase5_stack import load_logits

TFIDF = r"C:\Users\Anshuman Raj\CS781\phase5_runs\tfidf_lr_char"
TRAINDEV = r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-traindev\eff_traindev"
HIERDEV = r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-hierdev\eff_hierdev"
W384 = r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-w384ft\eff_w384ft"
OUT = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p5_sub\efficient_last_tfidf"


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def main():
    root = os.path.join(r"C:\Users\Anshuman Raj\CS781\phase5data", "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)
    names = {"tfidf": TFIDF, "traindev": TRAINDEV, "hierdev": HIERDEV, "w384": W384}
    dev, test = {}, {}
    for name, path in names.items():
        dev[name], test[name] = standardize(*load_logits(path, dev_ids, test_ids))

    best = (-1.0, None)
    for a in (0.6, 0.8, 1.0, 1.2, 1.5):
        for b in (0.0, 0.2, 0.4, 0.6):
            for c in (0.0, 0.15, 0.3):
                logits = dev["tfidf"] + a * dev["traindev"] + b * dev["hierdev"] + c * dev["w384"]
                value = threshold_report(gold, sigmoid(logits))["f1_per_label_crossval"]
                print(f"td {a} hd {b} w {c} {value:.4f}", flush=True)
                if value > best[0]:
                    best = (value, (a, b, c))
    print("best leaked", round(best[0], 4), best[1], flush=True)
    a, b, c = best[1]
    dev_probs = sigmoid(dev["tfidf"] + a * dev["traindev"] + b * dev["hierdev"] + c * dev["w384"])
    test_probs = sigmoid(test["tfidf"] + a * test["traindev"] + b * test["hierdev"] + c * test["w384"])
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
    report["leaked"] = True
    report["weights"] = {"tfidf": 1.0, "traindev": a, "hierdev": b, "w384": c}
    report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
    report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", os.path.join(OUT, "submission.zip"), round(report["f1_per_label_crossval"], 4), flush=True)


if __name__ == "__main__":
    main()
