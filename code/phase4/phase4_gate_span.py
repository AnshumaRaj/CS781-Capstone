"""Change the 0.6527 model only when the sentence model and the wide model agree."""
import json
import os
import zipfile

from phase4_own import macro_f1

DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
LONG = r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_long"
WIDE = r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_today\p4-partial-wide\p4-partial-wide"
SENT = r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_today\p4-partial-sent\p4-partial-sent"
DEST = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p4_sub\efficient_partial_span"


def load_map(folder, name):
    with open(os.path.join(folder, name), encoding="utf-8") as handle:
        rows = json.load(handle)
    return {str(row["id"]): float(row["score"]) for row in rows}


def choose(base, wide, sent):
    if abs(base - 0.5) < 0.08 and abs(wide - 0.5) > 0.15 and abs(sent - 0.5) > 0.15:
        wide_cls = 1 if wide >= 0.5 else 0
        sent_cls = 1 if sent >= 0.5 else 0
        if wide_cls == sent_cls:
            return sent_cls, True
    return (1 if base >= 0.5 else 0), False


def main():
    with open(DEV, encoding="utf-8") as handle:
        dev = json.load(handle)
    order = [str(row["id"]) for row in dev]
    gold = [int(row["label"]) for row in dev]
    base = load_map(LONG, "dev_scores.json")
    wide = load_map(WIDE, "dev_scores.json")
    sent = load_map(SENT, "dev_scores.json")
    pred = []
    fixed = broken = flips = 0
    for item, label in zip(order, gold):
        guess, flipped = choose(base[item], wide[item], sent[item])
        old = 1 if base[item] >= 0.5 else 0
        if flipped and guess != old:
            flips += 1
            if old == label and guess != label:
                broken += 1
            if old != label and guess == label:
                fixed += 1
        pred.append(guess)
    score = macro_f1(gold, pred)
    print("span dev", round(score, 4), "flips", flips, "fixed", fixed, "broken", broken, flush=True)
    if fixed <= broken:
        print("no zip", flush=True)
        return
    base_t = load_map(LONG, "test_scores.json")
    wide_t = load_map(WIDE, "test_scores.json")
    sent_t = load_map(SENT, "test_scores.json")
    with open(os.path.join(LONG, "test_scores.json"), encoding="utf-8") as handle:
        ids = [str(row["id"]) for row in json.load(handle)]
    predictions = []
    test_flips = 0
    for item in ids:
        guess, flipped = choose(base_t[item], wide_t[item], sent_t[item])
        old = 1 if base_t[item] >= 0.5 else 0
        if flipped and guess != old:
            test_flips += 1
        predictions.append({"id": item, "pred": int(guess)})
    os.makedirs(DEST, exist_ok=True)
    pred_path = os.path.join(DEST, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    zip_path = os.path.join(DEST, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("wrote", zip_path, "test_flips", test_flips, "rows", len(predictions), flush=True)


if __name__ == "__main__":
    main()
