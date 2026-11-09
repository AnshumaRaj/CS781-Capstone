"""Where outcome cues sit, using dev labels only."""
import json
import re

DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
POSITIVE = ("allowed", "allow", "aside", "quashed", "quash", "succeeds", "accepted", "unsustainable")
NEGATIVE = ("dismissed", "dismiss", "fails", "fail", "rightly", "merit", "upheld", "rejected", "interfere")


def words_of(text):
    return [re.sub(r"[^a-z]", "", word.lower()) for word in text.split()]


def hits(tokens, vocab):
    return [token for token in tokens if token in vocab]


def main():
    with open(DEV, encoding="utf-8") as handle:
        dev = json.load(handle)
    both = set(POSITIVE) | set(NEGATIVE)
    for n in (80, 200, 500):
        covered = agree = 0
        for row in dev:
            tail = words_of(row["text"])[-n:]
            pos = len(hits(tail, set(POSITIVE)))
            neg = len(hits(tail, set(NEGATIVE)))
            if pos == neg:
                continue
            covered += 1
            pred = 1 if pos > neg else 0
            agree += int(pred == int(row["label"]))
        print(f"last {n:4} covered {covered:4} accuracy {agree / max(covered, 1):.3f}", flush=True)
    only_early = 0
    for row in dev:
        tokens = words_of(row["text"])
        near = set(hits(tokens[-80:], both))
        early = set(hits(tokens[-500:-80], both))
        if early and not near:
            only_early += 1
    print("cue only in words 500..80", only_early, "of", len(dev), flush=True)


if __name__ == "__main__":
    main()
