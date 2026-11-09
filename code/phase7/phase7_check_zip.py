"""Check Phase 7 submission zips: root prediction.json, 7 unique pool ids per test query."""
import json
import os
import sys
import zipfile

TEST = r"C:\Users\Anshuman Raj\CS781\phase7data\Phase_7_PCR\Test\test.json"
CAND = r"C:\Users\Anshuman Raj\CS781\phase7data\Phase_7_PCR\Test\test_candidates.json"
K = 7


def main():
    with open(TEST, encoding="utf-8") as handle:
        queries = json.load(handle)
    with open(CAND, encoding="utf-8") as handle:
        pool = set(json.load(handle))
    expected = set(queries)
    for path in sys.argv[1:]:
        with zipfile.ZipFile(path) as bundle:
            names = bundle.namelist()
            payload = json.loads(bundle.read("prediction.json")) if "prediction.json" in names else {}
        bad = []
        if names != ["prediction.json"]:
            bad.append("zip names " + str(names))
        if set(payload) != expected:
            bad.append(f"ids missing={len(expected - set(payload))} extra={len(set(payload) - expected)}")
        for qid, vals in payload.items():
            if not isinstance(vals, list) or len(vals) != K:
                bad.append(f"{qid} len {0 if not isinstance(vals, list) else len(vals)}")
                continue
            if len(set(vals)) != K:
                bad.append(f"{qid} duplicates")
            if any(not isinstance(cid, str) or not cid.strip() or cid not in pool for cid in vals):
                bad.append(f"{qid} off-pool")
            if qid in vals:
                bad.append(f"{qid} self")
        print(path, "ok" if not bad else "BAD", "rows", len(payload), "issues", bad[:6], flush=True)


if __name__ == "__main__":
    main()
