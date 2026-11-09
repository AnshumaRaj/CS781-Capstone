import json
import re

DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
TRAIN = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\train.json"
CUES = (
    "allowed",
    "allow",
    "aside",
    "quashed",
    "dismissed",
    "dismiss",
    "fails",
    "fail",
    "rightly",
    "merit",
    "interfere",
)


def words(text, n):
    raw = text.lower().split()[-n:]
    return [re.sub(r"[^a-z]", "", word) for word in raw]


def rate(rows, n):
    print("window", n, "docs", len(rows))
    for cue in CUES:
        hit = [row for row in rows if cue in words(row["text"], n)]
        if not hit:
            print(f"  {cue:12} none")
            continue
        ones = sum(int(row["label"]) == 1 for row in hit)
        print(f"  {cue:12} {len(hit):5}  class1 {ones / len(hit):.3f}")


def main():
    with open(DEV, encoding="utf-8") as handle:
        dev = json.load(handle)
    with open(TRAIN, encoding="utf-8") as handle:
        train = json.load(handle)
    print("DEV")
    rate(dev, 120)
    print("TRAIN sample")
    rate(train[:8000], 120)


if __name__ == "__main__":
    main()
