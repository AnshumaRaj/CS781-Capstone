"""Keep the 0.6527 model's prediction unless the continuation is much more sure."""
import json
import os
import zipfile

LONG = r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_long"
MORE = r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_more"
DEST = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p4_sub\efficient_partial_gate"


def load_scores(folder, name):
    with open(os.path.join(folder, name), encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["id"]: float(row["score"]) for row in rows}, [row["id"] for row in rows]


def main():
    base, order = load_scores(MORE, "test_scores.json")
    # order must follow the long test file, which matches the test ids.
    _, order = load_scores(LONG, "test_scores.json")
    base = load_scores(LONG, "test_scores.json")[0]
    other = load_scores(MORE, "test_scores.json")[0]
    predictions = []
    flipped = 0
    for item in order:
        if abs(base[item] - 0.5) < 0.08 and abs(other[item] - 0.5) > 0.15:
            score = other[item]
            flipped += 1
        else:
            score = base[item]
        predictions.append({"id": item, "pred": int(score >= 0.5)})
    os.makedirs(DEST, exist_ok=True)
    pred_path = os.path.join(DEST, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    with zipfile.ZipFile(os.path.join(DEST, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    used = sorted({row["pred"] for row in predictions})
    print("flipped", flipped, "rows", len(predictions), "preds", used, flush=True)


if __name__ == "__main__":
    main()
