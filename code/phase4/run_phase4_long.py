"""Efficient models on the last 768 tokens, where earlier outcome lines still sit."""
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
    "--chunks", "3",
    "--chunk_tokens", "256",
    "--word_cap", "1000",
    "--balanced",
    "--rank", "16",
    "--alpha", "32",
]
JOBS = [
    (
        "efficient_partial_long",
        [
            "--arch", "efficient", "--method", "partial",
            "--unfreeze", "10", "--unfreeze_embed",
            "--epochs", "2", "--max_docs", "12000",
            "--lr", "3e-5", "--head_lr", "2e-4",
            "--warmup", "80", "--batch", "4",
        ],
    ),
    (
        "efficient_lorma_long",
        [
            "--arch", "efficient", "--method", "lorma",
            "--epochs", "2", "--max_docs", "16000",
            "--lr", "1e-4", "--head_lr", "3e-4",
            "--warmup", "80", "--batch", "8",
        ],
    ),
]


def store(name):
    run = os.path.join(RUNS, name)
    src = os.path.join(run, "submission.zip")
    with open(os.path.join(run, "metrics.json"), encoding="utf-8") as handle:
        metrics = json.load(handle)
    out = os.path.join(DEST, name)
    os.makedirs(out, exist_ok=True)
    shutil.copy2(src, os.path.join(out, "submission.zip"))
    with zipfile.ZipFile(src) as handle:
        pred = json.loads(handle.read("prediction.json"))
    print(
        "stored", name, "rows", len(pred),
        "preds", sorted({row["pred"] for row in pred}),
        "dev", round(metrics["chosen"], 4),
        flush=True,
    )


def main():
    for name, extra in JOBS:
        out = os.path.join(RUNS, name)
        cmd = ["python", "-u", PY, *extra, *COMMON, "--out", out]
        print("start", name, flush=True)
        code = subprocess.call(cmd, cwd=ROOT)
        print("exit", name, code, flush=True)
        if code != 0:
            raise SystemExit(code)
        store(name)


if __name__ == "__main__":
    main()
