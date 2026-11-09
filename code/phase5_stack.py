"""Stack Phase 5 runs: per-label logistic regression over all runs' logits, trained on dev with K-fold CV.

    python phase5_stack.py RUN_DIR [RUN_DIR ...] [--C 0.5] [--write OUT_DIR]

Out-of-fold dev probabilities give the honest score; test uses the average of the fold models.
"""
import argparse
import json
import os
import zipfile

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from phase5_ensemble import find_runs
from phase5_lsi import DEFAULT_DATA, apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report


def load_logits(folder, dev_ids, test_ids):
    out = []
    for split, order in (("dev", dev_ids), ("test", test_ids)):
        with open(os.path.join(folder, f"{split}_logits_ids.json"), encoding="utf-8") as handle:
            ids = json.load(handle)
        values = np.load(os.path.join(folder, f"{split}_logits.npy")).astype(np.float32)
        where = {item: index for index, item in enumerate(ids)}
        out.append(values[[where[item] for item in order]])
    return out


def fit_label(dev_x, test_x, target, folds, c_value):
    oof = np.zeros(len(dev_x), dtype=np.float32)
    test = np.zeros(len(test_x), dtype=np.float32)
    for tune, hold in folds:
        model = LogisticRegression(C=c_value, max_iter=500)
        model.fit(dev_x[tune], target[tune])
        oof[hold] = model.predict_proba(dev_x[hold])[:, 1]
        test += model.predict_proba(test_x)[:, 1] / len(folds)
    return oof, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--write", default="")
    args = parser.parse_args()

    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)

    folders = find_runs(args.runs)
    pairs = [load_logits(folder, dev_ids, test_ids) for folder in folders]
    dev_x = np.concatenate([np.clip(pair[0], -12, 12) for pair in pairs], axis=1)
    test_x = np.concatenate([np.clip(pair[1], -12, 12) for pair in pairs], axis=1)
    mean, std = dev_x.mean(0), dev_x.std(0) + 1e-6
    dev_x, test_x = (dev_x - mean) / std, (test_x - mean) / std
    print("runs", len(folders), "features", dev_x.shape[1], flush=True)

    folds = list(KFold(args.folds, shuffle=True, random_state=0).split(dev_x))
    results = Parallel(n_jobs=args.jobs)(
        delayed(fit_label)(dev_x, test_x, gold[:, label], folds, args.C) for label in range(gold.shape[1])
    )
    dev_probs = np.stack([item[0] for item in results], axis=1)
    test_probs = np.stack([item[1] for item in results], axis=1)

    report = threshold_report(gold, dev_probs)
    cuts = per_label_cuts(gold, dev_probs)
    report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
    report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
    report["C"] = args.C
    report["members"] = folders
    print("stack", json.dumps({key: value for key, value in report.items() if key != "members"}), flush=True)

    if args.write:
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
        print("wrote", os.path.join(args.write, "submission.zip"), flush=True)


if __name__ == "__main__":
    main()
