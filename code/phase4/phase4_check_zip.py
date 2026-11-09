"""Check a submission zip against the official test ids. Does not read labels."""
import json
import sys
import zipfile

TEST = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Test\test.json"


def main():
    with open(TEST, encoding="utf-8") as handle:
        expected = [str(row["id"]) for row in json.load(handle)]
    for path in sys.argv[1:]:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            rows = json.loads(bundle.read(names[0]))
        ids = [str(row["id"]) for row in rows]
        preds = [row["pred"] for row in rows]
        ok = (
            names == ["prediction.json"]
            and len(ids) == len(expected)
            and len(set(ids)) == len(expected)
            and set(ids) == set(expected)
            and set(preds) <= {0, 1}
            and all(isinstance(pred, int) and not isinstance(pred, bool) for pred in preds)
        )
        print(path, "ok" if ok else "BAD", "names", names, "rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
