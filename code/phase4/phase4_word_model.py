"""Regularized word model on the ending of each judgment. Dev only, never test labels."""
import json
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

TRAIN = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\train.json"
DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"


def ending(text, n):
    words = [re.sub(r"[^a-z]", "", word.lower()) for word in text.split()[-n:]]
    return " ".join(word for word in words if word)


def main():
    with open(TRAIN, encoding="utf-8") as handle:
        train = json.load(handle)
    with open(DEV, encoding="utf-8") as handle:
        dev = json.load(handle)
    rng = np.random.default_rng(42)
    rng.shuffle(train)
    train = train[:20000]
    for n in (80, 150, 400):
        train_text = [ending(row["text"], n) for row in train]
        dev_text = [ending(row["text"], n) for row in dev]
        train_y = np.array([int(row["label"]) for row in train])
        dev_y = np.array([int(row["label"]) for row in dev])
        vectorizer = CountVectorizer(min_df=20, ngram_range=(1, 2), max_features=12000)
        train_x = vectorizer.fit_transform(train_text)
        dev_x = vectorizer.transform(dev_text)
        model = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", solver="liblinear")
        model.fit(train_x, train_y)
        prob = model.predict_proba(dev_x)[:, 1]
        best_f1, best_cut = -1.0, 0.5
        for cut in np.linspace(0.2, 0.8, 25):
            pred = (prob >= cut).astype(int)
            score = f1_score(dev_y, pred, average="macro")
            if score > best_f1:
                best_f1, best_cut = score, cut
        argmax = f1_score(dev_y, model.predict(dev_x), average="macro")
        print(f"last {n:4} words  argmax {argmax:.4f}  tuned {best_f1:.4f} at {best_cut:.2f}", flush=True)


if __name__ == "__main__":
    main()
