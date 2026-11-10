"""Separate how much of the stack gain is threshold shift and how much is model combination."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from phase4_stack import BASE, DEV, TEST, f1_at, gather, logit

from phase4_own import macro_f1


def cv_preds(gold, feats, c_value, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros(len(gold))
    for train_idx, hold_idx in folds.split(feats, gold):
        model = LogisticRegression(C=c_value, max_iter=2000).fit(feats[train_idx], gold[train_idx])
        out[hold_idx] = model.predict_proba(feats[hold_idx])[:, 1]
    return out


def cv_threshold(gold, scores, seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(gold), dtype=int)
    cuts = np.arange(0.2, 0.81, 0.01)
    for train_idx, hold_idx in folds.split(scores.reshape(-1, 1), gold):
        best = max(cuts, key=lambda cut: f1_at(gold[train_idx], scores[train_idx], cut))
        pred[hold_idx] = scores[hold_idx] >= best
    return macro_f1(list(gold), list(pred))


def main():
    with open(DEV, encoding="utf-8") as handle:
        dev_rows = json.load(handle)
    with open(TEST, encoding="utf-8") as handle:
        test_ids = [str(row["id"]) for row in json.load(handle)]
    dev_ids = [str(row["id"]) for row in dev_rows]
    gold = np.array([int(row["label"]) for row in dev_rows])
    models = gather(dev_ids, test_ids)
    names = sorted(models, key=lambda name: -f1_at(gold, models[name][0]))
    base_dev, base_test = models[BASE]

    print("dev positive rate gold", gold.mean().round(3), flush=True)
    print("base positive rate dev", (base_dev >= 0.5).mean().round(3), "test", (base_test >= 0.5).mean().round(3), flush=True)
    print("base threshold-only cv", round(np.mean([cv_threshold(gold, base_dev, s) for s in range(3)]), 4), flush=True)
    mean_dev = np.mean([models[n][0] for n in names[:8]], axis=0)
    mean_test = np.mean([models[n][1] for n in names[:8]], axis=0)
    print("mean8 threshold-only cv", round(np.mean([cv_threshold(gold, mean_dev, s) for s in range(3)]), 4), flush=True)

    dev_feats = np.array([[logit(p) for p in models[n][0]] for n in names]).T
    test_feats = np.array([[logit(p) for p in models[n][1]] for n in names]).T
    model = LogisticRegression(C=0.2, max_iter=2000).fit(dev_feats, gold)
    stack_dev_cv = cv_preds(gold, dev_feats, 0.2, 0)
    stack_test = model.predict_proba(test_feats)[:, 1]
    print("stack intercept", round(float(model.intercept_[0]), 3), flush=True)
    print("stack positive rate dev(cv)", (stack_dev_cv >= 0.5).mean().round(3), "test", (stack_test >= 0.5).mean().round(3), flush=True)
    for name, weight in sorted(zip(names, model.coef_[0]), key=lambda item: -abs(item[1]))[:10]:
        print(f"  weight {name:30s} {weight:+.3f}", flush=True)

    # Same combination without the learned intercept effect: zero-mean the stack and cut at its own median.
    no_shift = []
    for seed in range(3):
        out = cv_preds(gold, dev_feats, 0.2, seed)
        no_shift.append(f1_at(gold, out, np.quantile(out, 1 - (base_dev >= 0.5).mean())))
    print("stack forced to base positive rate cv", round(float(np.mean(no_shift)), 4), flush=True)


if __name__ == "__main__":
    main()
