"""Blend TF-IDF with two groups of efficient-model runs: full fine-tunes and LoRMA runs.

    python phase5_groupblend.py [--write OUT_DIR]

Each run's logits are standardized, averaged within its group, and added to TF-IDF with a group weight.
Weights are chosen on the honest 2-fold per-label threshold score.
"""
import argparse
import json
import os
import zipfile

import numpy as np

from phase5_blend import standardize
from phase5_lsi import DEFAULT_DATA, apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report
from phase5_stack import load_logits

KAGGLE = r"C:\Users\Anshuman Raj\CS781\phase5_kaggle"
TFIDF = r"C:\Users\Anshuman Raj\CS781\phase5_runs\tfidf_lr_char"
FULL = ["p5-eff-part4\\eff_part4", "p5-eff-part10\\eff_part10", "p5-eff-partctx\\eff_partctx",
        "p5-eff-part10b\\eff_part10b", "p5-eff-parts2\\eff_parts2", "p5-eff-parthi\\eff_parthi"]
LORMA = ["p5-eff-ctx\\eff_ctx", "p5-eff-ctx2\\eff_ctx2", "p5-eff-long\\eff_long", "p5-eff-s2\\eff_s2",
         "p5-eff-w384\\eff_w384"]
LAPTOP = r"C:\Users\Anshuman Raj\CS781\phase5_runs\eff_ctx_laptop"


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", default="")
    args = parser.parse_args()

    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)

    def group(paths):
        loaded = [standardize(*load_logits(path, dev_ids, test_ids)) for path in paths if os.path.isfile(os.path.join(path, "dev_logits.npy"))]
        print("group size", len(loaded), flush=True)
        return np.mean([item[0] for item in loaded], axis=0), np.mean([item[1] for item in loaded], axis=0)

    base_dev, base_test = standardize(*load_logits(TFIDF, dev_ids, test_ids))
    full_dev, full_test = group([os.path.join(KAGGLE, item) for item in FULL])
    lorma_paths = [os.path.join(KAGGLE, item) for item in LORMA] + [LAPTOP]
    lorma_dev, lorma_test = group(lorma_paths)

    best = (0.0, None)
    for full_w in (0.3, 0.45, 0.6, 0.8):
        for lorma_w in (0.0, 0.1, 0.2, 0.3):
            value = threshold_report(gold, sigmoid(base_dev + full_w * full_dev + lorma_w * lorma_dev))["f1_per_label_crossval"]
            print(f"full {full_w} lorma {lorma_w} {value:.4f}", flush=True)
            best = max(best, (value, (full_w, lorma_w)))
    print("best", round(best[0], 4), best[1], flush=True)

    if args.write:
        full_w, lorma_w = best[1]
        dev_probs = sigmoid(base_dev + full_w * full_dev + lorma_w * lorma_dev)
        test_probs = sigmoid(base_test + full_w * full_test + lorma_w * lorma_test)
        report = threshold_report(gold, dev_probs)
        cuts = per_label_cuts(gold, dev_probs)
        report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
        report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
        report["weights"] = {"full": full_w, "lorma": lorma_w}
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
        with open(os.path.join(args.write, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print("wrote", os.path.join(args.write, "submission.zip"), round(report["f1_per_label_crossval"], 4), flush=True)


if __name__ == "__main__":
    main()
