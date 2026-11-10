"""Where the ending word model is wrong on dev. No test labels."""
import json

import joblib

from phase4_words import ending, word_probabilities

DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
MODEL = r"C:\Users\Anshuman Raj\CS781\phase4_runs\word_end.joblib"
CUES = ("allow", "allowed", "aside", "quash", "dismiss", "dismissed", "fail", "rightly", "merit")


def main():
    with open(DEV, encoding="utf-8") as handle:
        dev = json.load(handle)
    bundle = joblib.load(MODEL)
    probs = word_probabilities(bundle, [row["text"] for row in dev])
    wrong = []
    for row, prob in zip(dev, probs):
        pred = 1 if float(prob) >= 0.5 else 0
        gold = int(row["label"])
        if pred != gold:
            tail = ending(row["text"], 90)
            hits = [cue for cue in CUES if cue in tail.split()]
            wrong.append((gold, pred, float(prob), hits, tail[-280:]))
    print("errors", len(wrong), "of", len(dev), flush=True)
    for gold in (0, 1):
        group = [item for item in wrong if item[0] == gold]
        with_cue = sum(1 for item in group if item[3])
        print(f"gold {gold} errors {len(group)} with a cue {with_cue}", flush=True)
        for item in group[:6]:
            print("---", item[0], item[1], round(item[2], 3), item[3], flush=True)
            print(item[4], flush=True)


if __name__ == "__main__":
    main()
