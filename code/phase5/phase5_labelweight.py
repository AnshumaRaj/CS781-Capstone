"""Per-label blend weight on top of the global scaling used by phase5_blend.

Each label gets its own weight, chosen on one half of dev and scored on the other half.
"""
import json
import os
import zipfile

import numpy as np

from phase5_blend import standardize
from phase5_lsi import DEFAULT_DATA, apply_cuts, label_matrix, macro_f1, micro_f1, read_rows
from phase5_stack import load_logits

NEURAL = {
    "part4": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-part4\eff_part4",
    "ctx": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-ctx\eff_ctx",
    "long": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-long\eff_long",
    "s2": r"C:\Users\Anshuman Raj\CS781\phase5_kaggle\p5-eff-s2\eff_s2",
}
TFIDF = r"C:\Users\Anshuman Raj\CS781\phase5_runs\tfidf_lr_char"
WEIGHTS = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.1], dtype=np.float32)
CUTS = np.round(np.arange(0.05, 0.91, 0.05), 2).astype(np.float32)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def choose(gold, base, other):
    """gold, base, other: (n, labels). Returns per-label weight and probability cut."""
    chosen_w = np.zeros(gold.shape[1], dtype=np.float32)
    chosen_t = np.zeros(gold.shape[1], dtype=np.float32)
    for label in range(gold.shape[1]):
        logits = base[:, label][:, None] + other[:, label][:, None] * WEIGHTS[None, :]
        pred = sigmoid(logits)[:, :, None] >= CUTS[None, None, :]
        y = gold[:, label][:, None, None]
        tp = (pred * y).sum(0)
        fp = (pred * (1 - y)).sum(0)
        fn = ((1 - pred) * y).sum(0)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        flat = int(np.argmax(f1))
        weight_index, cut_index = divmod(flat, f1.shape[1])
        chosen_w[label] = WEIGHTS[weight_index]
        chosen_t[label] = CUTS[cut_index]
    return chosen_w, chosen_t


def score_split(gold, probs, cuts):
    pred = (probs >= cuts[None, :]).astype(np.float32)
    empty = pred.sum(1) == 0
    if empty.any():
        pred[empty, probs[empty].argmax(1)] = 1.0
    tp = (gold * pred).sum(0)
    fp = ((1 - gold) * pred).sum(0)
    fn = (gold * (1 - pred)).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(f1.mean())


def honest(gold, base, other):
    order = np.random.default_rng(0).permutation(len(gold))
    halves = [order[: len(order) // 2], order[len(order) // 2 :]]
    scores = []
    for tune, hold in (halves, halves[::-1]):
        weights, cuts = choose(gold[tune], base[tune], other[tune])
        probs = sigmoid(base[hold] + weights[None, :] * other[hold])
        scores.append(score_split(gold[hold], probs, cuts))
    return float(np.mean(scores))


def main():
    root = os.path.join(DEFAULT_DATA, "Phase_5_LSI")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), labeled=True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), labeled=False)
    dev_ids = [row["id"] for row in dev_rows]
    test_ids = [row["id"] for row in test_rows]
    gold = label_matrix(dev_rows)
    base_dev, base_test = standardize(*load_logits(TFIDF, dev_ids, test_ids))
    neural = {name: standardize(*load_logits(path, dev_ids, test_ids)) for name, path in NEURAL.items()}
    groups = {
        "four": ["part4", "ctx", "long", "s2"],
        "strong": ["part4", "ctx"],
        "part4": ["part4"],
    }
    best = (0.0, None)
    for name, members in groups.items():
        other_dev = np.mean([neural[item][0] for item in members], axis=0)
        other_test = np.mean([neural[item][1] for item in members], axis=0)
        value = honest(gold, base_dev, other_dev)
        print(name, round(value, 4), flush=True)
        if value > best[0]:
            best = (value, (other_dev, other_test, name))

    print("best", round(best[0], 4), best[1][2], flush=True)
    if best[0] <= 0.4205:
        return
    other_dev, other_test, name = best[1]
    weights, cuts = choose(gold, base_dev, other_dev)
    dev_probs = sigmoid(base_dev + weights[None, :] * other_dev)
    test_probs = sigmoid(base_test + weights[None, :] * other_test)
    out = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p5_sub\efficient_labelweight_tfidf"
    os.makedirs(out, exist_ok=True)
    pred = apply_cuts(test_probs, cuts)
    predictions = [
        {"id": item, "pred": [int(label) for label in np.flatnonzero(pred[index])]}
        for index, item in enumerate(test_ids)
    ]
    pred_path = os.path.join(out, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("wrote", os.path.join(out, "submission.zip"), "group", name,
          "optimistic", round(macro_f1(gold, apply_cuts(dev_probs, cuts)), 4),
          "micro", round(micro_f1(gold, apply_cuts(dev_probs, cuts)), 4), flush=True)


if __name__ == "__main__":
    main()
