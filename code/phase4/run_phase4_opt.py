"""Retrain efficient LoRMA and efficient partial on the decision ending."""
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
    "--tail_tokens", "192",
    "--word_cap", "280",
    "--balanced",
    "--rank", "16",
    "--alpha", "32",
]
JOBS = [
    (
        "efficient_lorma",
        [
            "--arch", "efficient", "--method", "lorma",
            "--epochs", "3", "--max_docs", "16000",
            "--lr", "1e-4", "--head_lr", "3e-4",
            "--warmup", "100", "--batch", "16",
        ],
    ),
    (
        "efficient_partial",
        [
            "--arch", "efficient", "--method", "partial",
            "--unfreeze", "10", "--unfreeze_embed",
            "--epochs", "2", "--max_docs", "12000",
            "--lr", "5e-5", "--head_lr", "2e-4",
            "--warmup", "80", "--batch", "4",
        ],
    ),
]


def store(name):
    run = os.path.join(RUNS, name + "_opt")
    src = os.path.join(run, "submission.zip")
    with open(os.path.join(run, "metrics.json"), encoding="utf-8") as handle:
        metrics = json.load(handle)
    out = os.path.join(DEST, name + "_opt")
    os.makedirs(out, exist_ok=True)
    shutil.copy2(src, os.path.join(out, "submission.zip"))
    with zipfile.ZipFile(src) as handle:
        pred = json.loads(handle.read("prediction.json"))
    preds = sorted({row["pred"] for row in pred})
    print("stored", name, "rows", len(pred), "preds", preds, "dev", round(metrics["chosen"], 4), flush=True)
    if metrics["chosen"] >= 0.60 and len(pred) == 1517 and preds == [0, 1]:
        live = os.path.join(DEST, name)
        os.makedirs(live, exist_ok=True)
        previous = os.path.join(live, "submission.zip")
        if os.path.isfile(previous):
            shutil.copy2(previous, os.path.join(live, "submission_prev.zip"))
        shutil.copy2(src, previous)
        print("replaced", live, flush=True)


def main():
    for name, extra in JOBS:
        out = os.path.join(RUNS, name + "_opt")
        cmd = ["python", "-u", PY, *extra, *COMMON, "--out", out]
        print("start", name, flush=True)
        code = subprocess.call(cmd, cwd=ROOT)
        print("exit", name, code, flush=True)
        if code != 0:
            raise SystemExit(code)
        store(name)


if __name__ == "__main__":
    main()
