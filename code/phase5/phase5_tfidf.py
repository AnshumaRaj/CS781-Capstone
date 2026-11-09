"""TF-IDF + per-label logistic regression for Phase 5. CPU only, no pretrained weights.

Writes dev/test logits in the same layout as phase5_lsi.py so phase5_ensemble.py can blend them.
"""
import argparse
import json
import os
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

from phase5_lsi import DEFAULT_DATA, LOCAL, label_matrix, read_rows, threshold_report


def fit_label(features, target, dev_x, test_x, c_value, folds=None, kind="lr"):
    if kind == "svm":
        model = LinearSVC(C=c_value, class_weight="balanced")
        model.fit(features.copy(), target)
        return model.decision_function(dev_x), model.decision_function(test_x), None
    model = LogisticRegression(C=c_value, solver="liblinear", class_weight="balanced", max_iter=200)
    model.fit(features, target)
    oof = None
    if folds:
        oof = np.zeros(features.shape[0], dtype=np.float32)
        for tune, hold in folds:
            part = LogisticRegression(C=c_value, solver="liblinear", class_weight="balanced", max_iter=200)
            part.fit(features[tune], target[tune])
            oof[hold] = part.decision_function(features[hold])
    return model.decision_function(dev_x), model.decision_function(test_x), oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(LOCAL, "phase5_runs", "tfidf_lr"))
    parser.add_argument("--word_features", type=int, default=300000)
    parser.add_argument("--char_features", type=int, default=200000)
    parser.add_argument("--C", type=float, default=4.0)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--oof", type=int, default=0, help="also write K-fold out-of-fold train logits")
    parser.add_argument("--model", choices=("lr", "svm"), default="lr")
    parser.add_argument("--word_ngram", type=int, default=2)
    parser.add_argument("--with_dev", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    train = read_rows(os.path.join(root, "Train", "train.json"), labeled=True)
    dev = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    texts = [[row["text"] for row in rows] for rows in (train, dev, test)]

    fit_texts = texts[0] + texts[1] if args.with_dev else texts[0]
    word = TfidfVectorizer(ngram_range=(1, args.word_ngram), max_features=args.word_features, min_df=3, max_df=0.9,
                           sublinear_tf=True, lowercase=True, dtype=np.float32)
    word.fit(fit_texts)
    parts = [word.transform(block) for block in texts]
    print("word tfidf", parts[0].shape, round(time.time() - started), "s", flush=True)
    if args.char_features:
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=args.char_features, min_df=5,
                               sublinear_tf=True, lowercase=True, dtype=np.float32)
        char.fit(fit_texts)
        extra = [char.transform(block) for block in texts]
        parts = [hstack([a, b]).tocsr() for a, b in zip(parts, extra)]
        print("char tfidf", extra[0].shape, round(time.time() - started), "s", flush=True)

    for matrix in parts:
        matrix.sum_duplicates()
        matrix.sort_indices()
    gold_train, gold_dev = label_matrix(train), label_matrix(dev)
    fit_x, fit_y = parts[0], gold_train
    if args.with_dev:
        fit_x = vstack([parts[0], parts[1]]).tocsr()
        fit_x.sum_duplicates()
        fit_x.sort_indices()
        fit_y = np.concatenate([gold_train, gold_dev], axis=0)
        print("fitting on train+dev", fit_x.shape, flush=True)
    folds = list(KFold(args.oof, shuffle=True, random_state=0).split(np.arange(len(fit_y)))) if args.oof else None
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fit_label)(fit_x, fit_y[:, label], parts[1], parts[2], args.C, folds, args.model)
        for label in range(gold_train.shape[1])
    )
    dev_logits = np.stack([item[0] for item in results], axis=1).astype(np.float32)
    test_logits = np.stack([item[1] for item in results], axis=1).astype(np.float32)
    if folds:
        train_oof = np.stack([item[2] for item in results], axis=1).astype(np.float32)
        np.save(os.path.join(args.out, "train_logits.npy"), train_oof)
        with open(os.path.join(args.out, "train_logits_ids.json"), "w", encoding="utf-8") as handle:
            json.dump([row["id"] for row in train], handle)
        oof_report = threshold_report(gold_train, 1.0 / (1.0 + np.exp(-train_oof)))
        print("train out-of-fold", json.dumps(oof_report), flush=True)

    report = threshold_report(gold_dev, 1.0 / (1.0 + np.exp(-dev_logits)))
    report["minutes"] = round((time.time() - started) / 60, 1)
    print("dev report", json.dumps(report), flush=True)
    np.save(os.path.join(args.out, "dev_logits.npy"), dev_logits.astype(np.float16))
    np.save(os.path.join(args.out, "test_logits.npy"), test_logits.astype(np.float16))
    for name, rows in (("dev", dev), ("test", test)):
        with open(os.path.join(args.out, f"{name}_logits_ids.json"), "w", encoding="utf-8") as handle:
            json.dump([row["id"] for row in rows], handle)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
