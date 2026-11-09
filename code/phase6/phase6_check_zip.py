"""Check Phase 6 submission zips: root prediction.json, one non-empty summary per test id."""
import json
import sys
import zipfile

TEST = r"C:\Users\Anshuman Raj\CS781\phase6data\Phase_6_Summ\Test\test.json"


def test_ids():
    with open(TEST, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [str(row["id"]) for row in payload]
    first = next(iter(payload.values()))
    if isinstance(first, dict):
        return [str(row["id"]) for row in payload.values()]
    return [str(key) for key in payload]


def read_pred(payload):
    if isinstance(payload, list):
        out = {}
        for row in payload:
            out[str(row["id"])] = str(row["summary"])
        return out
    return {str(key): str(value) for key, value in payload.items()}


def main():
    expected = test_ids()
    for path in sys.argv[1:]:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            payload = json.loads(bundle.read("prediction.json")) if "prediction.json" in names else {}
        pred = read_pred(payload)
        empty = sum(1 for value in pred.values() if not str(value).strip())
        ok = names == ["prediction.json"] and set(pred) == set(expected) and empty == 0
        lengths = [len(value.split()) for value in pred.values()] if pred else [0]
        print(path, "ok" if ok else "BAD", "names", names, "rows", len(pred),
              "empty", empty, "words", round(sum(lengths) / max(len(lengths), 1), 1), flush=True)


if __name__ == "__main__":
    main()
