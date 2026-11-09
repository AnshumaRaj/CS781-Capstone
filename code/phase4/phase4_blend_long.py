"""Blend the long efficient partial model with efficient LoRMA. Dev labels only."""
import json
import os
import zipfile

from phase4_own import macro_f1

RUNS = r"C:\Users\Anshuman Raj\CS781\phase4_runs"
DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
DEST = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p4_sub"


def load_scores(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["id"]: float(row["score"]) for row in rows}, [row["id"] for row in rows]


def f1_of(scores, order, gold, cut=0.5):
    pred = [1 if scores[item] >= cut else 0 for item in order]
    return macro_f1([gold[item] for item in order], pred)


def main():
    with open(DEV, encoding="utf-8") as handle:
        gold = {str(row["id"]): int(row["label"]) for row in json.load(handle)}
    names = ("efficient_lorma_opt", "efficient_partial_opt", "efficient_partial_long")
    maps = {}
    order = None
    for name in names:
        scores, ids = load_scores(os.path.join(RUNS, name, "dev_scores.json"))
        maps[name] = scores
        order = ids
        print(name, round(f1_of(scores, order, gold), 4), flush=True)
    pairs = {
        "lorma_long": ("efficient_lorma_opt", "efficient_partial_long"),
        "all_three": names,
    }
    test_maps = {}
    test_order = None
    for name in names:
        scores, ids = load_scores(os.path.join(RUNS, name, "test_scores.json"))
        test_maps[name] = scores
        test_order = ids
    for label, group in pairs.items():
        mixed = {item: sum(maps[name][item] for name in group) / len(group) for item in order}
        print(label, "dev", round(f1_of(mixed, order, gold), 4), flush=True)
        predictions = []
        for item in test_order:
            score = sum(test_maps[name][item] for name in group) / len(group)
            predictions.append({"id": item, "pred": int(score >= 0.5)})
        out_dir = os.path.join(DEST, "efficient_" + label)
        os.makedirs(out_dir, exist_ok=True)
        pred_path = os.path.join(out_dir, "prediction.json")
        with open(pred_path, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle)
        with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(pred_path, arcname="prediction.json")
        used = sorted({row["pred"] for row in predictions})
        print("zip", label, len(predictions), used, flush=True)


if __name__ == "__main__":
    main()
