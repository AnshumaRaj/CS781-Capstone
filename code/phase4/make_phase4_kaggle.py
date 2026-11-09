"""Build the Phase 4 Kaggle dataset folder and six notebooks."""
import json
import os
import shutil

CODE = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781"
LOCAL = r"C:\Users\Anshuman Raj\CS781"
DATA = os.path.join(LOCAL, "kaggle_phase4_data")
USER_JOBS = {
    "demonhunter24": ("efficient_lorma", "hybrid_adapter"),
    "bisendhritlahare": ("decoder_bitfit", "encdec_partial"),
    "anshuman1067": ("efficient_partial", "encdec_distill"),
}
JOBS = {
    "efficient_lorma": ("efficient", "lorma", []),
    "hybrid_adapter": ("hybrid", "adapter", []),
    "decoder_bitfit": ("decoder", "bitfit", []),
    "encdec_partial": ("encdec", "partial", []),
    "efficient_partial": ("efficient", "partial", []),
    "encdec_distill": ("encdec", "distill", []),
}

files = {
    "phase4_own.py": os.path.join(CODE, "phase4_own.py"),
    "phase1_models.py": os.path.join(CODE, "phase1_models.py"),
    "phase2_lner.py": os.path.join(CODE, "phase2_lner.py"),
    "legal_bpe.py": os.path.join(CODE, "legal_bpe.py"),
    "legal_bpe.json": os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\legal_bpe.json"),
    "efficient_model.pt": os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\model.pt"),
    "decoder_model.pt": os.path.join(LOCAL, r"local_weights\cs781_phase1_decoder\model.pt"),
    "hybrid_model.pt": os.path.join(LOCAL, r"local_weights\cs781_phase1_hybrid\model.pt"),
    "encdec_model.pt": os.path.join(LOCAL, r"encdec_run2\cs781_phase1_encdec\model.pt"),
}
os.makedirs(DATA, exist_ok=True)
for name, source in files.items():
    dest = os.path.join(DATA, name)
    if not os.path.isfile(dest) or os.path.getsize(dest) != os.path.getsize(source):
        shutil.copyfile(source, dest)
        print("copied", name, os.path.getsize(dest), flush=True)
    else:
        print("kept", name, os.path.getsize(dest), flush=True)

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
for name, dest in (
    ("train.json.gz", data / "Train" / "train.json"),
    ("dev.json.gz", data / "Train" / "dev.json"),
    ("test.json.gz", data / "Test" / "test.json"),
):
    with gzip.open(src / name, "rb") as inp, open(dest, "wb") as out:
        shutil.copyfileobj(inp, out)
base = [
    sys.executable, "-u", str(work / "phase4_own.py"),
    "--bpe", str(work / "legal_bpe.json"),
    "--data_dir", str(work / "phase4data"),
    "--epochs", "2", "--max_docs", "3000",
]
runs = {runs}
for arch, method, out_name, extra in runs:
    cmd = base + ["--arch", arch, "--method", method, "--weights", str(work / f"{arch}_model.pt"), "--out", str(work / out_name)] + extra
    print(" ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(work))
    if code != 0:
        raise SystemExit(code)
'''

for user, job_names in USER_JOBS.items():
    for job in job_names:
        arch, method, extra = JOBS[job]
        runs = [(arch, method, job, extra)]
        if job == "encdec_distill":
            runs = [
                ("efficient", "lorma", "teacher_lorma", ["--epochs", "1"]),
                (
                    "encdec",
                    "distill",
                    job,
                    ["--teacher", "/kaggle/working/teacher_lorma/best.pt", "--weights", "/kaggle/working/encdec_model.pt"],
                ),
            ]
        folder = os.path.join(LOCAL, f"kaggle_phase4_{job}")
        os.makedirs(folder, exist_ok=True)
        code_file = f"phase4-{job.replace('_', '-')}.ipynb"
        source = CELL.replace("{runs}", repr(runs))
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
        print("kernel", folder, flush=True)
