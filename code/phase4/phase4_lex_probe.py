"""Ceiling of a train-only lexical model on the judgment ending. Dev labels only."""
import json
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

TRAIN = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\train.json"
DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"


def ending(text, n):
    words = [re.sub(r"[^a-z]", "", word.lower()) for word in str(text).split()[-n:]]
    return " ".join(word for word in words if word)


def score_split(name, train_text, dev_text, train_y, dev_y, vectorizer):
    train_x = vectorizer.fit_transform(train_text)
    dev_x = vectorizer.transform(dev_text)
    model = LogisticRegression(C=1.0, max_iter=250, class_weight="balanced", solver="liblinear")
    model.fit(train_x, train_y)
    pred = model.predict(dev_x)
    prob = model.predict_proba(dev_x)[:, 1]
    argmax = f1_score(dev_y, pred, average="macro")
    best_f1, best_cut = argmax, 0.5
    for cut in np.linspace(0.3, 0.7, 17):
        tuned = f1_score(dev_y, (prob >= cut).astype(int), average="macro")
        if tuned > best_f1:
            best_f1, best_cut = tuned, float(cut)
    print(f"{name} argmax {argmax:.4f} tuned {best_f1:.4f} at {best_cut:.2f}", flush=True)


def main():
    with open(TRAIN, encoding="utf-8") as handle:
        train = json.load(handle)
    with open(DEV, encoding="utf-8") as handle:
        dev = json.load(handle)
    rng = np.random.default_rng(42)
    rng.shuffle(train)
    train = train[:25000]
    train_y = np.array([int(row["label"]) for row in train])
    dev_y = np.array([int(row["label"]) for row in dev])
    for n, ngram, features in (
        (60, (1, 2), 12000),
        (80, (1, 2), 12000),
        (80, (1, 3), 20000),
        (120, (1, 2), 12000),
    ):
        score_split(
            f"last {n} word{ngram}",
            [ending(row["text"], n) for row in train],
            [ending(row["text"], n) for row in dev],
            train_y,
            dev_y,
            CountVectorizer(min_df=10, ngram_range=ngram, max_features=features),
        )
    score_split(
        "last 80 char",
        [ending(row["text"], 80) for row in train],
        [ending(row["text"], 80) for row in dev],
        train_y,
        dev_y,
        CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=10, max_features=12000),
    )


if __name__ == "__main__":
    main()
