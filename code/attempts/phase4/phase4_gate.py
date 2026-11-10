"""See whether other partial models can correct the 0.6527 model's uncertain dev cases."""
import json
import os

from phase4_own import macro_f1

DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
LONG = r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_long"
TODAY = r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_today"
OTHERS = {
    "more": r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_more",
    "wide": os.path.join(TODAY, "p4-partial-wide", "p4-partial-wide"),
    "sent": os.path.join(TODAY, "p4-partial-sent", "p4-partial-sent"),
    "s7": os.path.join(TODAY, "p4-partial-s7", "p4-partial-s7"),
}


def load_map(folder):
    with open(os.path.join(folder, "dev_scores.json"), encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["id"]: float(row["score"]) for row in rows}


def f1(gold, pred):
    return macro_f1(gold, pred)


def main():
    with open(DEV, encoding="utf-8") as handle:
        dev = json.load(handle)
    order = [str(row["id"]) for row in dev]
    gold = [int(row["label"]) for row in dev]
    base = load_map(LONG)
    base_pred = [1 if base[item] >= 0.5 else 0 for item in order]
    print("long", round(f1(gold, base_pred), 4), flush=True)
    for name, folder in OTHERS.items():
        other = load_map(folder)
        pred = [1 if other[item] >= 0.5 else 0 for item in order]
        print(name, round(f1(gold, pred), 4), flush=True)
        wrong = [index for index, (g, p) in enumerate(zip(gold, base_pred)) if g != p]
        fixed = 0
        broken = 0
        for index in range(len(order)):
            item = order[index]
            unsure = abs(base[item] - 0.5) < 0.08
            sure = abs(other[item] - 0.5) > 0.15
            if not (unsure and sure):
                continue
            guess = 1 if other[item] >= 0.5 else 0
            if base_pred[index] == gold[index] and guess != gold[index]:
                broken += 1
            if base_pred[index] != gold[index] and guess == gold[index]:
                fixed += 1
        gated = []
        for index, item in enumerate(order):
            if abs(base[item] - 0.5) < 0.08 and abs(other[item] - 0.5) > 0.15:
                gated.append(1 if other[item] >= 0.5 else 0)
            else:
                gated.append(base_pred[index])
        print(
            name,
            "fixed", fixed,
            "broken", broken,
            "gated", round(f1(gold, gated), 4),
            flush=True,
        )


if __name__ == "__main__":
    main()
