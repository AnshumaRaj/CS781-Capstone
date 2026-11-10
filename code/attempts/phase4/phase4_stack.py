"""Combine every finished efficient model. Scores are judged with 5-fold cross-validation on dev."""
import json
import math
import os
import sys
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from phase4_own import macro_f1

DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
TEST = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Test\test.json"
ROOTS = [
    r"C:\Users\Anshuman Raj\CS781\phase4_runs",
    r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_today",
    r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_round3",
    r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_fast",
]
SKIP = {"efficient_partial_newdocs", "efficient_partial_swa", "p4-partial-s7"}
SUB = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p4_sub"
BASE = "efficient_partial_long"


def load(path):
    with open(path, encoding="utf-8") as handle:
        return {str(row["id"]): float(row["score"]) for row in json.load(handle)}


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def gather(dev_ids, test_ids):
    models = {}
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for folder, _, files in os.walk(root):
            if "dev_scores.json" not in files or "test_scores.json" not in files:
                continue
            name = os.path.basename(folder)
            if name in SKIP or name in models:
                continue
            if "efficient" not in folder and "p4-" not in name:
                continue
            dev = load(os.path.join(folder, "dev_scores.json"))
            test = load(os.path.join(folder, "test_scores.json"))
            if set(dev_ids) - set(dev) or set(test_ids) - set(test):
                continue
            models[name] = (
                np.array([dev[item] for item in dev_ids]),
                np.array([test[item] for item in test_ids]),
            )
    return models


def f1_at(gold, scores, cut=0.5):
    return macro_f1(list(gold), [int(score >= cut) for score in scores])


def cv_stack(gold, feats, c_value, seeds=(0, 1, 2)):
    results = []
    for seed in seeds:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        out = np.zeros(len(gold))
        for train_idx, hold_idx in folds.split(feats, gold):
            model = LogisticRegression(C=c_value, max_iter=2000)
            model.fit(feats[train_idx], gold[train_idx])
            out[hold_idx] = model.predict_proba(feats[hold_idx])[:, 1]
        results.append(f1_at(gold, out))
    return float(np.mean(results))


def write_zip(name, test_ids, preds):
    dest = os.path.join(SUB, name)
    os.makedirs(dest, exist_ok=True)
    rows = [{"id": item, "pred": int(pred)} for item, pred in zip(test_ids, preds)]
    pred_path = os.path.join(dest, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    with zipfile.ZipFile(os.path.join(dest, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("wrote", dest, "positives", int(sum(preds)), "of", len(preds), flush=True)


def main():
    write = "--write" in sys.argv
    with open(DEV, encoding="utf-8") as handle:
        dev_rows = json.load(handle)
    with open(TEST, encoding="utf-8") as handle:
        test_ids = [str(row["id"]) for row in json.load(handle)]
    dev_ids = [str(row["id"]) for row in dev_rows]
    gold = np.array([int(row["label"]) for row in dev_rows])
    models = gather(dev_ids, test_ids)
    ranked = sorted(models, key=lambda name: -f1_at(gold, models[name][0]))
    for name in ranked:
        print(f"{name:34s} dev {f1_at(gold, models[name][0]):.4f}", flush=True)
    base_dev, base_test = models[BASE]
    print("base", BASE, round(f1_at(gold, base_dev), 4), flush=True)

    best = {}
    for top in (3, 5, 8, 12, len(ranked)):
        names = ranked[:top]
        if BASE not in names:
            names = [BASE] + names[: top - 1]
        dev_mean = np.mean([models[n][0] for n in names], axis=0)
        test_mean = np.mean([models[n][1] for n in names], axis=0)
        score = f1_at(gold, dev_mean)
        print(f"mean top{top:2d} dev {score:.4f}", flush=True)
        best[f"mean{top}"] = (score, test_mean >= 0.5)

    names = ranked
    dev_feats = np.array([[logit(p) for p in models[n][0]] for n in names]).T
    test_feats = np.array([[logit(p) for p in models[n][1]] for n in names]).T
    for c_value in (0.01, 0.05, 0.2):
        score = cv_stack(gold, dev_feats, c_value)
        model = LogisticRegression(C=c_value, max_iter=2000).fit(dev_feats, gold)
        test_pred = model.predict_proba(test_feats)[:, 1] >= 0.5
        print(f"stack C={c_value} cv dev {score:.4f}", flush=True)
        best[f"stack{c_value}"] = (score, test_pred)

    winner = "stack0.05"
    score, preds = best[winner]
    base_pred = base_test >= 0.5
    print("winner", winner, round(score, 4), "test changes vs 0.6527 file", int((preds != base_pred).sum()), flush=True)
    if write and score >= f1_at(gold, base_dev) + 0.015:
        name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "efficient_final_ensemble"
        write_zip(name, test_ids, preds.astype(int))


if __name__ == "__main__":
    main()
