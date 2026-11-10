"""Profile the Phase 5 data: label counts, label cardinality, document length, field names."""
import collections
import json
import os
import re

import numpy as np

ROOT = r"C:\Users\Anshuman Raj\CS781\phase5data\Phase_5_LSI"


def load(split, name):
    with open(os.path.join(ROOT, split, name), encoding="utf-8") as handle:
        return json.load(handle)


def main():
    train = load("Train", "train.json")
    dev = load("Train", "dev.json")
    test = load("Test", "test.json")
    print("sizes", len(train), len(dev), len(test), flush=True)
    print("train keys", list(train[0].keys()), "test keys", list(test[0].keys()), flush=True)
    for key, value in train[0].items():
        text = str(value)
        print(" ", key, type(value).__name__, text[:300].replace("\n", " | "), flush=True)

    for name, rows in (("train", train), ("dev", dev)):
        counts = collections.Counter(label for row in rows for label in row["labels"])
        card = [len(row["labels"]) for row in rows]
        print(
            name, "labels", len(counts), "min id", min(counts), "max id", max(counts),
            "per doc mean", round(float(np.mean(card)), 2), "max", max(card),
            "docs with 1 label", sum(c == 1 for c in card), flush=True,
        )
        freq = sorted(counts.values(), reverse=True)
        print("  top 10 counts", freq[:10], "bottom 10", freq[-10:], flush=True)
        print("  labels with <=5 docs", sum(c <= 5 for c in freq), "<=20", sum(c <= 20 for c in freq), flush=True)

    train_labels = set(label for row in train for label in row["labels"])
    dev_labels = set(label for row in dev for label in row["labels"])
    print("dev labels missing from train", sorted(dev_labels - train_labels)[:30], len(dev_labels - train_labels), flush=True)
    print("train labels missing from dev", len(train_labels - dev_labels), flush=True)

    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        words = [len(str(row["text"]).split()) for row in rows]
        print(name, "words p10/p50/p90/max", np.percentile(words, [10, 50, 90]).astype(int).tolist(), max(words), flush=True)

    pairs = collections.Counter()
    for row in train:
        labels = sorted(row["labels"])
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs[(labels[i], labels[j])] += 1
    print("top co-occurring pairs", pairs.most_common(10), flush=True)

    sample = str(train[0]["text"])
    hits = re.findall(r"(?:section|sec\.|s\.|article|art\.)\s*\d+[a-z]?", sample, flags=re.I)
    print("statute mentions in first doc", hits[:20], flush=True)
    print("first doc labels", train[0]["labels"], flush=True)


if __name__ == "__main__":
    main()
