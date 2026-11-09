"""Inspect Phase 7 PCR files. No model training."""
import json
import os
import statistics
from collections import Counter

ROOT = r"C:\Users\Anshuman Raj\CS781\phase7data\Phase_7_PCR"


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def show_obj(name, obj, n=2):
    print(f"\n== {name} type={type(obj).__name__}", flush=True)
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("n", len(keys), "first_ids", keys[:5], flush=True)
        sample = obj[keys[0]]
        print("value_type", type(sample).__name__, flush=True)
        if isinstance(sample, dict):
            print("value_keys", sorted(sample.keys()), flush=True)
            for key, value in sample.items():
                if isinstance(value, str):
                    print(f"  {key} str_len={len(value)} words={len(value.split())} head={value[:160]!r}", flush=True)
                elif isinstance(value, list):
                    print(f"  {key} list_n={len(value)} head={value[:8]!r}", flush=True)
                else:
                    print(f"  {key}", type(value).__name__, repr(value)[:120], flush=True)
        elif isinstance(sample, list):
            print("list_n", len(sample), "elem", type(sample[0]).__name__ if sample else None, flush=True)
        elif isinstance(sample, str):
            print("str words", len(sample.split()), "head", sample[:160], flush=True)
    elif isinstance(obj, list):
        print("n", len(obj), "elem_type", type(obj[0]).__name__, flush=True)
        print("elem", obj[0] if not isinstance(obj[0], dict) else {k: (type(v).__name__, str(v)[:80]) for k, v in obj[0].items()}, flush=True)


def main():
    files = {
        "train": os.path.join(ROOT, "Train", "train.json"),
        "train_cand": os.path.join(ROOT, "Train", "train_candidates.json"),
        "dev": os.path.join(ROOT, "Dev", "dev.json"),
        "dev_cand": os.path.join(ROOT, "Dev", "dev_candidates.json"),
        "test": os.path.join(ROOT, "Test", "test.json"),
        "test_cand": os.path.join(ROOT, "Test", "test_candidates.json"),
    }
    for name, path in files.items():
        print(name, os.path.getsize(path), flush=True)
        obj = load(path)
        show_obj(name, obj)

    train = load(files["train"])
    train_c = load(files["train_cand"])
    rel_n = []
    missing = 0
    for qid, row in train.items():
        if isinstance(row, dict):
            rel = row.get("relevant_candidates") or row.get("relevant") or []
            text = row.get("text") or row.get("query") or ""
        else:
            rel, text = [], ""
        rel_n.append(len(rel))
        for cid in rel:
            if isinstance(train_c, dict) and cid not in train_c:
                missing += 1
    print("\ntrain rel per query mean", round(statistics.mean(rel_n), 3), "p50", sorted(rel_n)[len(rel_n)//2],
          "min", min(rel_n), "max", max(rel_n), "missing_in_pool", missing, flush=True)
    if isinstance(train, dict):
        qwords = [len(str((v.get("text") if isinstance(v, dict) else v) or "").split()) for v in train.values()]
        print("query words p50", sorted(qwords)[len(qwords)//2], "p90", sorted(qwords)[int(0.9*len(qwords))], flush=True)
    if isinstance(train_c, dict):
        cwords = []
        first = next(iter(train_c.values()))
        if isinstance(first, dict):
            cwords = [len(str(v.get("text") or v.get("document") or "").split()) for v in train_c.values()]
        elif isinstance(first, str):
            cwords = [len(v.split()) for v in train_c.values()]
        if cwords:
            print("cand words p50", sorted(cwords)[len(cwords)//2], "p90", sorted(cwords)[int(0.9*len(cwords))], flush=True)


if __name__ == "__main__":
    main()
