"""Finer global blend of the finished runs. One scale per model, as in phase5_blend."""
import numpy as np

from phase5_blend import standardize
from phase5_lsi import DEFAULT_DATA, label_matrix, read_rows, threshold_report
from phase5_stack import load_logits

PATHS = {
    "tfidf": r"C:\Users\Anshuman Raj\CS781\phase5_runs\tfidf_lr_char",
    "part4": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-part4\eff_part4",
    "ctx": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-ctx\eff_ctx",
    "long": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-long\eff_long",
    "s2": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-s2\eff_s2",
    "w384": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-w384\eff_w384",
}


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def main():
    root = __import__("os").path.join(DEFAULT_DATA, "Phase_5_LSI")
    dev_rows = read_rows(__import__("os").path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(__import__("os").path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)
    dev = {name: standardize(*load_logits(path, dev_ids, test_ids))[0] for name, path in PATHS.items()}
    strong = 0.5 * (dev["part4"] + dev["ctx"])
    rest = (dev["long"] + dev["s2"] + dev["w384"]) / 3
    best = (0.0, None)
    for strong_w in (0.15, 0.25, 0.35, 0.45):
        for rest_w in (0.0, 0.1, 0.2, 0.3):
            probs = sigmoid(dev["tfidf"] + strong_w * strong + rest_w * rest)
            value = threshold_report(gold, probs)["f1_per_label_crossval"]
            print(f"strong {strong_w} rest {rest_w} {value:.4f}", flush=True)
            best = max(best, (value, (strong_w, rest_w)))
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
