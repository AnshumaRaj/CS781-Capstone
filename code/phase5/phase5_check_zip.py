"""Check Phase 5 submission zips against the official test ids. Does not read labels."""
import json
import sys
import zipfile

TEST = r"C:\Users\Anshuman Raj\CS781\phase5data\Phase_5_LSI\Test\test.json"
LABELS = 100


def valid_pred(pred):
    return (
        isinstance(pred, list)
        and all(isinstance(label, int) and not isinstance(label, bool) for label in pred)
        and all(0 <= label < LABELS for label in pred)
        and len(set(pred)) == len(pred)
    )


def main():
    with open(TEST, encoding="utf-8") as handle:
        test = json.load(handle)
    expected = [row["id"] for row in test]
    for path in sys.argv[1:]:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            rows = json.loads(bundle.read(names[0]))
        ids = [row["id"] for row in rows]
        preds = [row["pred"] for row in rows]
        ok = (
            names == ["prediction.json"]
            and len(ids) == len(expected)
            and len(set(ids)) == len(ids)
            and set(ids) == set(expected)
            and all(valid_pred(pred) for pred in preds)
        )
        empty = sum(1 for pred in preds if not pred)
        mean = sum(len(pred) for pred in preds) / max(len(preds), 1)
        print(path, "ok" if ok else "BAD", "names", names, "rows", len(rows),
              "labels/doc", round(mean, 2), "empty", empty, flush=True)


if __name__ == "__main__":
    main()
