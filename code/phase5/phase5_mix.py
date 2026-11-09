"""Per-label blend and a small stack of the finished Phase 5 runs.

Weights and the stack are judged by the honest 2-fold per-label threshold score.
"""
import json
import os
import zipfile

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from phase5_stack import load_logits
from phase5_lsi import (
    DEFAULT_DATA,
    apply_cuts,
    label_matrix,
    macro_f1,
    micro_f1,
    per_label_cuts,
    read_rows,
    threshold_report,
)

RUNS = {
    "tfidf": r"C:\Users\Anshuman Raj\CS781\phase5_runs\tfidf_lr_char",
    "part4": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-part4\eff_part4",
    "ctx": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-ctx\eff_ctx",
    "long": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-long\eff_long",
    "s2": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-s2\eff_s2",
    "w384": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-w384\eff_w384",
    "partial": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-kd\eff_partial",
}


def scale(dev, test):
    mean, std = dev.mean(0), dev.std(0) + 1e-6
    return (dev - mean) / std, (test - mean) / std


def partners_of(gold, count=2):
    present = gold.sum(0)
    joint = gold.T @ gold
    score = joint / np.maximum(present[:, None], 1)
    np.fill_diagonal(score, 0)
    return np.argsort(-score, axis=1)[:, :count]


def fit_label(dev_x, test_x, target, folds, c_value):
    oof = np.zeros(len(dev_x), dtype=np.float32)
    test = np.zeros(len(test_x), dtype=np.float32)
    for tune, hold in folds:
        if target[tune].min() == target[tune].max():
            oof[hold] = target[tune].mean()
            test += target[tune].mean() / len(folds)
            continue
        model = LogisticRegression(C=c_value, max_iter=400)
        model.fit(dev_x[tune], target[tune])
        oof[hold] = model.predict_proba(dev_x[hold])[:, 1]
        test += model.predict_proba(test_x)[:, 1] / len(folds)
    return oof, test


def write_zip(path, test_ids, test_probs, gold, dev_probs, note):
    os.makedirs(path, exist_ok=True)
    cuts = per_label_cuts(gold, dev_probs)
    pred = apply_cuts(test_probs, cuts)
    predictions = [
        {"id": item, "pred": [int(label) for label in np.flatnonzero(pred[index])]}
        for index, item in enumerate(test_ids)
    ]
    pred_path = os.path.join(path, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    with zipfile.ZipFile(os.path.join(path, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    report = threshold_report(gold, dev_probs)
    report["micro_f1_per_label"] = micro_f1(gold, apply_cuts(dev_probs, cuts))
    report["f1_per_label_on_dev_optimistic"] = macro_f1(gold, apply_cuts(dev_probs, cuts))
    report["note"] = note
    with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", os.path.join(path, "submission.zip"), round(report["f1_per_label_crossval"], 4), flush=True)


def main():
    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    train_rows = read_rows(os.path.join(root, "Train", "train.json"), labeled=True)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)
    mates = partners_of(label_matrix(train_rows))

    scaled = {}
    for name, folder in RUNS.items():
        dev, test = load_logits(folder, dev_ids, test_ids)
        scaled[name] = scale(dev, test)
        print("loaded", name, flush=True)

    neural = ["part4", "ctx", "long", "s2", "w384", "partial"]
    base_dev, base_test = scaled["tfidf"]
    best = (0.0, None, None, None)
    for names, weight in (
        (["part4"], 0.5),
        (["part4", "ctx"], 0.5),
        (["part4", "ctx", "long", "s2"], 0.5),
        (neural, 0.35),
        (neural, 0.5),
        (neural, 0.7),
    ):
        other_dev = np.mean([scaled[name][0] for name in names], axis=0)
        other_test = np.mean([scaled[name][1] for name in names], axis=0)
        dev_probs = 1 / (1 + np.exp(-(base_dev + weight * other_dev)))
        value = threshold_report(gold, dev_probs)["f1_per_label_crossval"]
        print(f"blend {names} w {weight} {value:.4f}", flush=True)
        if value > best[0]:
            test_probs = 1 / (1 + np.exp(-(base_test + weight * other_test)))
            best = (value, dev_probs, test_probs, f"blend {names} w {weight}")

    folds = list(KFold(5, shuffle=True, random_state=0).split(gold))
    for names, c_value, use_mates in (
        (["tfidf", "part4", "ctx", "long", "s2"], 0.5, False),
        (["tfidf", "part4", "ctx", "long", "s2", "w384", "partial"], 0.3, False),
        (["tfidf", "part4", "ctx", "long", "s2"], 0.5, True),
        (["tfidf", "part4", "ctx", "long", "s2", "w384", "partial"], 0.3, True),
    ):
        blocks = [scaled[name][0] for name in names]
        test_blocks = [scaled[name][1] for name in names]
        dev_x = np.stack(blocks, axis=2)
        test_x = np.stack(test_blocks, axis=2)
        if use_mates:
            extra, extra_test = [], []
            for mate in range(mates.shape[1]):
                index = mates[:, mate]
                extra.append(dev_x[np.arange(len(dev_x))[:, None], index])
                extra_test.append(test_x[np.arange(len(test_x))[:, None], index])
            dev_x = np.concatenate([dev_x] + extra, axis=2)
            test_x = np.concatenate([test_x] + extra_test, axis=2)
        results = Parallel(n_jobs=4)(
            delayed(fit_label)(dev_x[:, label], test_x[:, label], gold[:, label], folds, c_value)
            for label in range(gold.shape[1])
        )
        dev_probs = np.stack([item[0] for item in results], axis=1)
        value = threshold_report(gold, dev_probs)["f1_per_label_crossval"]
        print(f"stack {names} mates {use_mates} C {c_value} {value:.4f}", flush=True)
        if value > best[0]:
            best = (value, dev_probs, np.stack([item[1] for item in results], axis=1),
                    f"stack {names} mates {use_mates} C {c_value}")

    print("best", round(best[0], 4), best[3], flush=True)
    if best[0] > 0.4205:
        write_zip(
            r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p5_sub\efficient_mix_tfidf",
            test_ids, best[2], gold, best[1], best[3],
        )


if __name__ == "__main__":
    main()
