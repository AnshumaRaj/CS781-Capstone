import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781")
RUNS = Path(r"C:\Users\Anshuman Raj\CS781\phase4_runs")
DEST = ROOT / "p4_sub"
PY = str(ROOT / "phase4_own.py")
COMMON = [
    "--epochs", "2",
    "--max_docs", "3000",
    "--chunks", "6",
    "--chunk_tokens", "128",
]
JOBS = [
    ("decoder_bitfit", ["--arch", "decoder", "--method", "bitfit"]),
    ("encdec_partial", ["--arch", "encdec", "--method", "partial"]),
    ("efficient_partial", ["--arch", "efficient", "--method", "partial"]),
    (
        "encdec_distill",
        [
            "--arch", "encdec",
            "--method", "distill",
            "--student_method", "lora",
            "--teacher", str(RUNS / "efficient_lorma" / "best.pt"),
        ],
    ),
]


def store(name):
    src = RUNS / name / "submission.zip"
    out = DEST / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out / "submission.zip")
    with zipfile.ZipFile(src) as handle:
        names = handle.namelist()
        pred = json.loads(handle.read("prediction.json"))
    preds = sorted({row["pred"] for row in pred})
    ids = {row["id"] for row in pred}
    print("stored", name, "files", names, "rows", len(pred), "unique", len(ids), "preds", preds, flush=True)
    if names != ["prediction.json"] or len(pred) != 1517 or len(ids) != 1517 or not set(preds) <= {0, 1}:
        raise SystemExit("bad zip " + name)


def main():
    for name, extra in JOBS:
        out = RUNS / name
        if (out / "submission.zip").exists():
            print("already done", name, flush=True)
        else:
            cmd = ["python", "-u", PY, *extra, *COMMON, "--out", str(out)]
            print("start", name, flush=True)
            code = subprocess.call(cmd, cwd=str(ROOT))
            print("exit", name, code, flush=True)
            if code != 0:
                raise SystemExit(code)
        store(name)
    print("threshold sweep", flush=True)
    code = subprocess.call(["python", "-u", str(ROOT / "phase4_threshold.py")], cwd=str(ROOT))
    print("threshold exit", code, flush=True)


if __name__ == "__main__":
    main()
