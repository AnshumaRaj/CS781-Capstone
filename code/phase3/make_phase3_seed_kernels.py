"""Kaggle notebooks for extra Phase 3 seeds.

Usage: python make_phase3_seed_kernels.py efficient:lorma:7 hybrid:lora:13 decoder:ia3:7 efficient:lorma:7:full

A fourth field "full" trains on train plus dev for a fixed number of epochs.
"v2" uses the stronger head: 2-layer BiLSTM 384, domain embed, sentence attention.
Any other fourth field passes trainer options, e.g. efficient:lorma:7:chunk=64+rank=16.
"""
import json
import os
import sys

LOCAL = r"C:\Users\Anshuman Raj\CS781"
USER = "demonhunter24"
RUNS = [tuple(item.split(":")) for item in sys.argv[1:]] or [("efficient", "lorma", "7"), ("efficient", "lorma", "13")]

CELL = '''import shutil, subprocess, sys
from pathlib import Path

root = Path("/kaggle/input")
src = sorted(root.rglob("phase3_rr_own.py"))[0].parent
weights = src / "{arch}_model.pt"
if not weights.is_file():
    weights = sorted(root.rglob("model.pt"))[0]
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
    "--seed", "{seed}",
    "--weights", str(weights),
    "--bpe", str(src / "legal_bpe.json"),
    "--data_dir", str(work / "phase3data"),
    "--out", str(work / "{arch}_{method}_s{seed}"),
    "--epochs", "{epochs}",
    "--patience", "{patience}",
] + {extra}
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd, cwd=str(work)))
'''

for arch, method, seed, *mode in RUNS:
    full = mode == ["full"]
    epochs, patience = ("8", "3") if arch == "efficient" else ("16", "4")
    extra = []
    tag = f"s{seed}"
    if full:
        epochs = "5" if arch == "efficient" else "8"
        extra = ["--fixed", "--include_dev"]
        tag += "-full"
    elif mode and mode[0] in ("v2", "v2b"):
        epochs, patience = "12", "5"
        if mode[0] == "v2":
            extra = [
                "--head_hidden", "384",
                "--head_layers", "2",
                "--domain_dim", "32",
                "--attn_heads", "4",
                "--epochs", "12",
                "--patience", "5",
            ]
        else:
            # Milder head: domain + 2-layer BiLSTM, no self-attention.
            extra = [
                "--head_hidden", "256",
                "--head_layers", "2",
                "--domain_dim", "32",
                "--attn_heads", "0",
                "--head_lr", "5e-4",
                "--epochs", "12",
                "--patience", "5",
            ]
        tag += f"-{mode[0]}"
    elif mode:
        for option in mode[0].split("+"):
            key, value = option.split("=")
            extra += [f"--{key}", value]
            tag += f"-{key.replace('_', '')}{value.replace('.', 'p')}"
    name = f"phase3-{arch}-{method}-{tag}"
    folder = os.path.join(LOCAL, f"kaggle_phase3_{arch}_{method}_{tag.replace('-', '_')}")
    os.makedirs(folder, exist_ok=True)
    code_file = f"{name}.ipynb"
    source = CELL.format(arch=arch, method=method, seed=seed, epochs=epochs, patience=patience, extra=repr(extra))
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(os.path.join(folder, code_file), "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1)
    sources = [f"{USER}/phase3-rr"] + ([f"{USER}/phase2-efficient"] if arch == "efficient" else [])
    metadata = {
        "id": f"{USER}/{name}",
        "title": name.replace("-", " "),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": sources,
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("kernel", folder, flush=True)
