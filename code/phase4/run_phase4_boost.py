import json
import os
import shutil
import subprocess
import zipfile

ROOT = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781"
RUNS = r"C:\Users\Anshuman Raj\CS781\phase4_runs"
DEST = os.path.join(ROOT, "p4_sub")
PY = os.path.join(ROOT, "phase4_own.py")
COMMON = [
    "--focus", "tail",
    "--cover", "tail",
    "--chunks", "4",
    "--chunk_tokens", "256",
    "--word_cap", "1200",
    "--epochs", "3",
    "--max_docs", "8000",
    "--word_feature",
    "--rank", "16",
    "--alpha", "32",
]
JOBS = [
    ("efficient_lorma", ["--arch", "efficient", "--method", "lorma"]),
    ("efficient_partial", ["--arch", "efficient", "--method", "partial", "--unfreeze", "6"]),
]


def store(name):
    src = os.path.join(RUNS, name + "_v2", "submission.zip")
    out = os.path.join(DEST, name)
    os.makedirs(out, exist_ok=True)
    shutil.copy2(src, os.path.join(out, "submission.zip"))
    with zipfile.ZipFile(src) as handle:
        pred = json.loads(handle.read("prediction.json"))
    preds = sorted({row["pred"] for row in pred})
    print("stored", name, "rows", len(pred), "preds", preds, flush=True)


def main():
    for name, extra in JOBS:
        out = os.path.join(RUNS, name + "_v2")
        cmd = ["python", "-u", PY, *extra, *COMMON, "--out", out]
        print("start", name, flush=True)
        code = subprocess.call(cmd, cwd=ROOT)
        print("exit", name, code, flush=True)
        if code != 0:
            raise SystemExit(code)
        store(name)


if __name__ == "__main__":
    main()
