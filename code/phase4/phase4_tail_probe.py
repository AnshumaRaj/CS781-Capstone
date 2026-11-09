"""See whether the last words of a judgment already separate the two outcomes."""
import json
from collections import Counter

import numpy as np

TRAIN = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\train.json"
DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
TAIL = 200
MAX_DOCS = 12000


def load(path, limit=None):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if limit:
        rows = rows[:limit]
    return rows


def tail_words(text):
    return text.split()[-TAIL:]


def macro_f1(gold, pred):
    scores = []
    for label in (0, 1):
        tp = int(np.sum((gold == label) & (pred == label)))
        fp = int(np.sum((gold != label) & (pred == label)))
        fn = int(np.sum((gold == label) & (pred != label)))
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / 2


def main():
    train = load(TRAIN)
    rng = np.random.default_rng(42)
    rng.shuffle(train)
    train = train[:MAX_DOCS]
    dev = load(DEV)
    counts = Counter()
    for row in train:
        counts.update(set(tail_words(row["text"])))
    vocab = [word for word, count in counts.most_common(6000) if count >= 40]
    index = {word: i for i, word in enumerate(vocab)}
    print("vocab", len(vocab), flush=True)

    def matrix(rows):
        x = np.zeros((len(rows), len(vocab)), dtype=np.float32)
        y = np.zeros(len(rows), dtype=np.int64)
        for row_index, row in enumerate(rows):
            y[row_index] = int(row["label"])
            for word in set(tail_words(row["text"])):
                column = index.get(word)
                if column is not None:
                    x[row_index, column] = 1.0
        return x, y

    train_x, train_y = matrix(train)
    dev_x, dev_y = matrix(dev)
    weight = np.array(
        [len(train_y) / (2 * max(1, np.sum(train_y == label))) for label in (0, 1)],
        dtype=np.float32,
    )
    coef = np.zeros((len(vocab), 2), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    lr = 0.5
    for epoch in range(1, 9):
        order = np.random.permutation(len(train_y))
        for start in range(0, len(order), 256):
            batch = order[start : start + 256]
            scores = train_x[batch] @ coef + bias
            scores = scores - scores.max(axis=1, keepdims=True)
            prob = np.exp(scores)
            prob /= prob.sum(axis=1, keepdims=True)
            gold = np.zeros_like(prob)
            gold[np.arange(len(batch)), train_y[batch]] = 1.0
            sample_w = weight[train_y[batch]][:, None]
            grad = (prob - gold) * sample_w
            coef -= lr * (train_x[batch].T @ grad) / len(batch)
            bias -= lr * grad.mean(axis=0)
        pred = (dev_x @ coef + bias).argmax(axis=1)
        print(f"epoch {epoch} dev macro-F1 {macro_f1(dev_y, pred):.4f}", flush=True)
    strength = coef[:, 1] - coef[:, 0]
    cue = np.argsort(np.abs(strength))[-30:]
    for column in cue[::-1]:
        print(f"{strength[column]:+.3f} {vocab[column]}", flush=True)


if __name__ == "__main__":
    main()
