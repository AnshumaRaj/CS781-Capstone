"""Build the Kaggle dataset and the two training notebooks for Phase 3."""
import json
import os
import shutil

CODE = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781"
LOCAL = r"C:\Users\Anshuman Raj\CS781"
DATA = os.path.join(LOCAL, r"phase3data\Phase_3_RR")
DATASET = os.path.join(LOCAL, "kaggle_phase3_data")
USER = "demonhunter24"
SLUG = "phase3-rr"

files = {
    "phase3_rr_own.py": os.path.join(CODE, "phase3_rr_own.py"),
    "phase2_lner.py": os.path.join(CODE, "phase2_lner.py"),
    "phase1_models.py": os.path.join(CODE, "phase1_models.py"),
    "legal_bpe.py": os.path.join(CODE, "legal_bpe.py"),
    "legal_bpe.json": os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\legal_bpe.json"),
    "decoder_model.pt": os.path.join(LOCAL, r"local_weights\cs781_phase1_decoder\model.pt"),
    "hybrid_model.pt": os.path.join(LOCAL, r"local_weights\cs781_phase1_hybrid\model.pt"),
}
for name in ("label_vocab.json", "CL_train.json", "IT_train.json", "CL_dev.json", "IT_dev.json"):
    files[name] = os.path.join(DATA, "Train", name)
for name in ("CL_test.json", "IT_test.json"):
    files[name] = os.path.join(DATA, "Test", name)

os.makedirs(DATASET, exist_ok=True)
for name, source in files.items():
    dest = os.path.join(DATASET, name)
    if not os.path.isfile(dest) or os.path.getsize(dest) != os.path.getsize(source) or name.endswith(".py"):
        shutil.copyfile(source, dest)
    print("dataset", name, os.path.getsize(dest), flush=True)
with open(os.path.join(DATASET, "dataset-metadata.json"), "w", encoding="utf-8") as handle:
    json.dump({"title": SLUG, "id": f"{USER}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}, handle, indent=2)

CELL = '''import shutil, subprocess, sys
from pathlib import Path

found = sorted(Path("/kaggle/input").rglob("phase3_rr_own.py"))
if not found:
    raise SystemExit("phase3_rr_own.py was not mounted")
src = found[0].parent
work = Path("/kaggle/working")
for name in ("phase3_rr_own.py", "phase2_lner.py", "phase1_models.py", "legal_bpe.py"):
    shutil.copy(src / name, work / name)
data = work / "phase3data" / "Phase_3_RR"
for split, names in {{"Train": ["label_vocab.json", "CL_train.json", "IT_train.json", "CL_dev.json", "IT_dev.json"],
                     "Test": ["CL_test.json", "IT_test.json"]}}.items():
    (data / split).mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy(src / name, data / split / name)

cmd = [
    sys.executable, "-u", str(work / "phase3_rr_own.py"),
    "--arch", "{arch}",
    "--method", "{method}",
    "--weights", str(src / "{arch}_model.pt"),
    "--bpe", str(src / "legal_bpe.json"),
    "--data_dir", str(work / "phase3data"),
    "--out", str(work / "{arch}_{method}"),
    "--epochs", "16",
    "--patience", "4",
]
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, cwd=str(work)))
'''

for arch, method in (("decoder", "ia3"), ("hybrid", "lora")):
    folder = os.path.join(LOCAL, f"kaggle_phase3_{arch}_{method}")
    os.makedirs(folder, exist_ok=True)
    code_file = f"phase3-{arch}-{method}.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": CELL.format(arch=arch, method=method),
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(os.path.join(folder, code_file), "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1)
    metadata = {
        "id": f"{USER}/phase3-{arch}-{method}",
        "title": f"phase3 {arch} {method}",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{USER}/{SLUG}"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("kernel", folder, flush=True)
