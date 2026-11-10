"""Choose the best efficient partial file from today's runs. Dev labels only."""
import json
import os
import zipfile

from phase4_own import macro_f1

DEV = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
DEST = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p4_sub"
RUNS = {
    "laptop_more": r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_more",
}
TODAY = r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_today"
for slug in ("p4-partial-s7", "p4-partial-sent", "p4-partial-c4", "p4-partial-s3", "p4-partial-s11", "p4-partial-wide"):
    RUNS[slug] = os.path.join(TODAY, slug, slug)


def load_scores(folder, name):
    path = os.path.join(folder, name)
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["id"]: float(row["score"]) for row in rows}, [row["id"] for row in rows]


def f1_of(scores, order, gold):
    pred = [1 if scores[item] >= 0.5 else 0 for item in order]
    return macro_f1([gold[item] for item in order], pred)


def write_zip(name, order, scores):
    out_dir = os.path.join(DEST, name)
    os.makedirs(out_dir, exist_ok=True)
    predictions = [{"id": item, "pred": int(scores[item] >= 0.5)} for item in order]
    pred_path = os.path.join(out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    zip_path = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    used = sorted({row["pred"] for row in predictions})
    print("zip", name, len(predictions), used, flush=True)


def main():
    with open(DEV, encoding="utf-8") as handle:
        gold = {str(row["id"]): int(row["label"]) for row in json.load(handle)}
    dev_maps = {}
    test_maps = {}
    dev_order = None
    test_order = None
    for name, folder in RUNS.items():
        metrics_path = os.path.join(folder, "metrics.json")
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        dev_scores, dev_order = load_scores(folder, "dev_scores.json")
        test_scores, test_order = load_scores(folder, "test_scores.json")
        dev_maps[name] = dev_scores
        test_maps[name] = test_scores
        print(name, "argmax", round(metrics["argmax"], 4), "recomputed", round(f1_of(dev_scores, dev_order, gold), 4), flush=True)
    ranked = sorted(dev_maps, key=lambda name: f1_of(dev_maps[name], dev_order, gold), reverse=True)
    best_name = ranked[0]
    best_f1 = f1_of(dev_maps[best_name], dev_order, gold)
    print("best single", best_name, round(best_f1, 4), flush=True)
    strong = [name for name in ranked if f1_of(dev_maps[name], dev_order, gold) >= 0.60]
    if len(strong) >= 2:
        mixed = {
            item: sum(dev_maps[name][item] for name in strong) / len(strong)
            for item in dev_order
        }
        mixed_f1 = f1_of(mixed, dev_order, gold)
        print("average of", strong, round(mixed_f1, 4), flush=True)
        if mixed_f1 >= best_f1 + 0.01:
            test_mixed = {
                item: sum(test_maps[name][item] for name in strong) / len(strong)
                for item in test_order
            }
            write_zip("efficient_partial_best", test_order, test_mixed)
            print("chose ensemble", round(mixed_f1, 4), flush=True)
            return
    write_zip("efficient_partial_best", test_order, test_maps[best_name])
    print("chose", best_name, round(best_f1, 4), flush=True)


if __name__ == "__main__":
    main()
