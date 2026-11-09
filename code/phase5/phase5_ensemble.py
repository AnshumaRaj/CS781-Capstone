"""Average the probabilities of several Phase 5 runs and write a submission.

    python phase5_ensemble.py RUN_DIR [RUN_DIR ...] [--greedy] [--write OUT_DIR]

Each RUN_DIR must hold dev_logits.npy, test_logits.npy and their *_ids.json files.
Scores are the honest 2-fold per-label threshold estimate on dev.
"""
import argparse
import json
import os
import zipfile

import numpy as np

from phase5_lsi import DEFAULT_DATA, apply_cuts, label_matrix, macro_f1, micro_f1, per_label_cuts, read_rows, threshold_report


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def load_run(folder, dev_ids, test_ids):
    probs = {}
    for split, order in (("dev", dev_ids), ("test", test_ids)):
        with open(os.path.join(folder, f"{split}_logits_ids.json"), encoding="utf-8") as handle:
            ids = json.load(handle)
        values = np.load(os.path.join(folder, f"{split}_logits.npy")).astype(np.float32)
        where = {item: index for index, item in enumerate(ids)}
        probs[split] = sigmoid(values[[where[item] for item in order]])
    return probs


def find_runs(paths):
    found = []
    for path in paths:
        for root, _, files in os.walk(path):
            if "dev_logits.npy" in files and "test_logits.npy" in files:
                found.append(root)
    return sorted(set(found))


def score(gold, probs):
    return threshold_report(gold, probs)["f1_per_label_crossval"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--greedy", action="store_true", help="forward-select runs (with repeats) by crossval F1")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--write", default="")
    args = parser.parse_args()

    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)

    folders = find_runs(args.runs)
    runs = {folder: load_run(folder, dev_ids, test_ids) for folder in folders}
    singles = {folder: score(gold, run["dev"]) for folder, run in runs.items()}
    for folder, value in sorted(singles.items(), key=lambda item: -item[1]):
        print(f"{value:.4f}  {folder}", flush=True)

    if args.greedy:
        chosen = [max(singles, key=singles.get)]
        best = singles[chosen[0]]
        for _ in range(args.rounds):
            trials = {}
            for folder in folders:
                pool = chosen + [folder]
                trials[folder] = score(gold, np.mean([runs[item]["dev"] for item in pool], axis=0))
            pick = max(trials, key=trials.get)
            if trials[pick] <= best + 1e-4:
                break
            chosen.append(pick)
            best = trials[pick]
            print(f"add {pick} -> {best:.4f}", flush=True)
    else:
        chosen = folders

    dev_probs = np.mean([runs[item]["dev"] for item in chosen], axis=0)
    test_probs = np.mean([runs[item]["test"] for item in chosen], axis=0)
    report = threshold_report(gold, dev_probs)
    cuts = per_label_cuts(gold, dev_probs)
    report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
    report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
    report["members"] = chosen
    print("ensemble", json.dumps(report, indent=2), flush=True)

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
