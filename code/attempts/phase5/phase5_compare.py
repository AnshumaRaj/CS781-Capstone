"""Compare blends on the honest per-label threshold score averaged over several random dev splits."""
import os

import numpy as np

from phase5_blend import standardize
from phase5_groupblend import FULL, KAGGLE, LORMA, TFIDF
from phase5_lsi import DEFAULT_DATA, label_matrix, read_rows, threshold_report
from phase5_stack import load_logits

SEEDS = range(5)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def robust(gold, probs):
    return float(np.mean([threshold_report(gold, probs, seed=seed)["f1_per_label_crossval"] for seed in SEEDS]))


def main():
    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)

    def dev_of(path):
        return standardize(*load_logits(path, dev_ids, test_ids))[0]

    tfidf = dev_of(TFIDF)
    runs = {item.split("\\")[0]: dev_of(os.path.join(KAGGLE, item)) for item in FULL + LORMA}
    old = tfidf + 0.45 * 0.5 * (runs["p5-eff-part4"] + runs["p5-eff-ctx"]) \
        + 0.1 * (runs["p5-eff-long"] + runs["p5-eff-s2"] + runs["p5-eff-w384"]) / 3
    print("current zip blend", round(robust(gold, sigmoid(old)), 4), flush=True)

    top_full = ["p5-eff-part4", "p5-eff-part10", "p5-eff-partctx", "p5-eff-part10b"]
    full = np.mean([runs[name] for name in top_full], axis=0)
    full = full / full.std()
    lorma = np.mean([runs[name] for name in ("p5-eff-ctx", "p5-eff-ctx2", "p5-eff-long", "p5-eff-s2", "p5-eff-w384")], axis=0)
    lorma = lorma / lorma.std()
    for full_w, lorma_w in ((0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.4, 0.2), (0.5, 0.0), (0.6, 0.1)):
        value = robust(gold, sigmoid(tfidf + full_w * full + lorma_w * lorma))
        print(f"top4 full {full_w} lorma {lorma_w} {value:.4f}", flush=True)


if __name__ == "__main__":
    main()
