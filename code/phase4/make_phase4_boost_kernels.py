import json
import os

LOCAL = r"C:\Users\Anshuman Raj\CS781"
CELL = r'''import gzip, shutil, subprocess, sys
from pathlib import Path
root = Path("/kaggle/input")
src = sorted(root.rglob("phase4_own.py"))[0].parent
work = Path("/kaggle/working")
for name in ("phase4_own.py", "phase1_models.py", "phase2_lner.py", "legal_bpe.py", "legal_bpe.json"):
    shutil.copy(src / name, work / name)
for path in src.glob("*_model.pt"):
    shutil.copy(path, work / path.name)
data = work / "phase4data" / "Phase_4_CJPE"
(data / "Train").mkdir(parents=True, exist_ok=True)
(data / "Test").mkdir(parents=True, exist_ok=True)
pairs = (
    ("train.json", data / "Train" / "train.json"),
    ("dev.json", data / "Train" / "dev.json"),
    ("test.json", data / "Test" / "test.json"),
)
for name, dest in pairs:
    packed = src / (name + ".gz")
    plain = src / name
    if packed.exists():
        with gzip.open(packed, "rb") as inp, open(dest, "wb") as out:
            shutil.copyfileobj(inp, out)
    elif plain.exists():
        shutil.copy(plain, dest)
    else:
        raise SystemExit("missing " + name)
base = [
    sys.executable, "-u", str(work / "phase4_own.py"),
    "--bpe", str(work / "legal_bpe.json"),
    "--data_dir", str(work / "phase4data"),
    "--epochs", "4", "--max_docs", "16000",
    "--chunks", "6", "--chunk_tokens", "128",
    "--rank", "16", "--alpha", "32",
]
runs = {runs}
for arch, method, out_name, extra in runs:
    cmd = base + ["--arch", arch, "--method", method, "--weights", str(work / f"{arch}_model.pt"), "--out", str(work / out_name)] + extra
    print(" ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(work))
    if code != 0:
        raise SystemExit(code)
'''

JOBS = [
    ("demonhunter24", "efficient_lorma_long", "efficient", "lorma", "42"),
    ("demonhunter24", "efficient_lorma_s7", "efficient", "lorma", "7"),
    ("anshuman1067", "efficient_partial_long", "efficient", "partial", "42"),
    ("anshuman1067", "efficient_partial_s7", "efficient", "partial", "7"),
]

for user, job, arch, method, seed in JOBS:
    folder = os.path.join(LOCAL, f"kaggle_phase4_{job}")
    os.makedirs(folder, exist_ok=True)
    runs = [(arch, method, job, ["--seed", seed])]
    source = CELL.replace("{runs}", repr(runs))
    code_file = f"phase4-{job.replace('_', '-')}.ipynb"
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(os.path.join(folder, code_file), "w", encoding="utf-8") as handle:
        json.dump(notebook, handle)
    metadata = {
        "id": f"{user}/phase4-{job.replace('_', '-')}",
        "title": f"phase4 {job.replace('_', ' ')}",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{user}/phase4-cjpe"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("wrote", folder)
