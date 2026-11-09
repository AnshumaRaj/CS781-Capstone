"""Outcome score from the last words of a judgment. Trained on the train split only."""
import os
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

TAIL_WORDS = 80


def ending(text, n=TAIL_WORDS):
    words = [re.sub(r"[^a-z]", "", word.lower()) for word in str(text).split()[-n:]]
    return " ".join(word for word in words if word)


def fit_word_model(rows, path):
    if os.path.isfile(path):
        return joblib.load(path)
    rng = np.random.default_rng(42)
    order = np.arange(len(rows))
    rng.shuffle(order)
    chosen = [rows[int(index)] for index in order[:20000]]
    texts = [ending(row["text"]) for row in chosen]
    labels = np.array([int(row["label"]) for row in chosen])
    vectorizer = CountVectorizer(min_df=15, ngram_range=(1, 2), max_features=8000)
    features = vectorizer.fit_transform(texts)
    model = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced", solver="liblinear")
    model.fit(features, labels)
    bundle = {"vectorizer": vectorizer, "model": model}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)
    print("saved word model", path, flush=True)
    return bundle


def word_probabilities(bundle, texts):
    features = bundle["vectorizer"].transform([ending(text) for text in texts])
    return bundle["model"].predict_proba(features)[:, 1].astype(np.float32)
