"""Average the two efficient models. Dev labels only; test file is ids and scores."""
import json
import os
import zipfile

from phase4_own import macro_f1

RUNS = r"C:\Users\Anshuman Raj\CS781\phase4_runs"
DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
DEST = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p4_sub"
NAMES = ("efficient_lorma_opt", "efficient_partial_opt")


def load_scores(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["id"]: float(row["score"]) for row in rows}, [row["id"] for row in rows]


def main():
    with open(DEV, encoding="utf-8") as handle:
        gold = {str(row["id"]): int(row["label"]) for row in json.load(handle)}
    maps = []
    order = None
    for name in NAMES:
        scores, ids = load_scores(os.path.join(RUNS, name, "dev_scores.json"))
        maps.append(scores)
        order = ids
        pred = [1 if scores[item] >= 0.5 else 0 for item in order]
        print(name, "dev", round(macro_f1([gold[item] for item in order], pred), 4), flush=True)
    mixed = []
    labels = []
    for item in order:
        mixed.append(0.5 * (maps[0][item] + maps[1][item]))
        labels.append(gold[item])
    pred = [1 if score >= 0.5 else 0 for score in mixed]
    print("average dev", round(macro_f1(labels, pred), 4), flush=True)
    test_maps = []
    test_order = None
    for name in NAMES:
        scores, ids = load_scores(os.path.join(RUNS, name, "test_scores.json"))
        test_maps.append(scores)
        test_order = ids
    out_dir = os.path.join(DEST, "efficient_ensemble")
    os.makedirs(out_dir, exist_ok=True)
    predictions = [
        {"id": item, "pred": int(0.5 * (test_maps[0][item] + test_maps[1][item]) >= 0.5)}
        for item in test_order
    ]
    pred_path = os.path.join(out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("zip", out_dir, "rows", len(predictions), flush=True)


if __name__ == "__main__":
    main()
