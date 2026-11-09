"""Use the 512-token model only where the 0.6527 model is unsure."""
import json
import os
import zipfile

from phase4_own import macro_f1

DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
LONG = r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_long"
OTHER = r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_round3\p4-partial-t512"
DEST = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p4_sub\efficient_partial_t512gate"


def load_map(folder, name):
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        path = os.path.join(folder, os.path.basename(folder), name)
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return {str(row["id"]): float(row["score"]) for row in rows}


def main():
    with open(DEV, encoding="utf-8") as handle:
        dev = json.load(handle)
    order = [str(row["id"]) for row in dev]
    gold = [int(row["label"]) for row in dev]
    base = load_map(LONG, "dev_scores.json")
    other = load_map(OTHER, "dev_scores.json")
    pred = []
    fixed = broken = flips = 0
    for item, label in zip(order, gold):
        old = 1 if base[item] >= 0.5 else 0
        if abs(base[item] - 0.5) < 0.08 and abs(other[item] - 0.5) > 0.15:
            guess = 1 if other[item] >= 0.5 else 0
        else:
            guess = old
        if guess != old:
            flips += 1
            if old == label and guess != label:
                broken += 1
            if old != label and guess == label:
                fixed += 1
        pred.append(guess)
    score = macro_f1(gold, pred)
    print("t512 gate", round(score, 4), "flips", flips, "fixed", fixed, "broken", broken, flush=True)
    if score < 0.630 or fixed <= broken * 2:
        print("no zip", flush=True)
        return
    base_t = load_map(LONG, "test_scores.json")
    other_t = load_map(OTHER, "test_scores.json")
    with open(os.path.join(LONG, "test_scores.json"), encoding="utf-8") as handle:
        ids = [str(row["id"]) for row in json.load(handle)]
    predictions = []
    test_flips = 0
    for item in ids:
        old = 1 if base_t[item] >= 0.5 else 0
        if abs(base_t[item] - 0.5) < 0.08 and abs(other_t[item] - 0.5) > 0.15:
            guess = 1 if other_t[item] >= 0.5 else 0
        else:
            guess = old
        if guess != old:
            test_flips += 1
        predictions.append({"id": item, "pred": int(guess)})
    os.makedirs(DEST, exist_ok=True)
    pred_path = os.path.join(DEST, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    zip_path = os.path.join(DEST, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("wrote", zip_path, "test_flips", test_flips, flush=True)


if __name__ == "__main__":
    main()
