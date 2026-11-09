"""Six efficient-partial kernels. The notebook carries the current trainer."""
import base64
import json
import os

LOCAL = r"C:\Users\Anshuman Raj\CS781"
SCRIPT = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\phase4_own.py"
WORDS = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\phase4_words.py"
with open(SCRIPT, "rb") as handle:
    PAYLOAD = base64.b64encode(handle.read()).decode("ascii")
with open(WORDS, "rb") as handle:
    WORDS_PAYLOAD = base64.b64encode(handle.read()).decode("ascii")

CELL = r'''import base64, gzip, shutil, subprocess, sys
from pathlib import Path
work = Path("/kaggle/working")
(work / "phase4_own.py").write_bytes(base64.b64decode("{payload}"))
(work / "phase4_words.py").write_bytes(base64.b64decode("{words}"))
src = sorted(Path("/kaggle/input").rglob("efficient_model.pt"))[0].parent
for name in ("phase1_models.py", "phase2_lner.py", "legal_bpe.py", "legal_bpe.json", "efficient_model.pt"):
    shutil.copy(src / name, work / name)
data = work / "phase4data" / "Phase_4_CJPE"
(data / "Train").mkdir(parents=True, exist_ok=True)
(data / "Test").mkdir(parents=True, exist_ok=True)
for name, dest in (
    ("train.json", data / "Train" / "train.json"),
    ("dev.json", data / "Train" / "dev.json"),
    ("test.json", data / "Test" / "test.json"),
):
    packed, plain = src / (name + ".gz"), src / name
    if packed.exists():
        with gzip.open(packed, "rb") as inp, open(dest, "wb") as out:
            shutil.copyfileobj(inp, out)
    elif plain.exists():
        shutil.copy(plain, dest)
    else:
        raise SystemExit("missing " + name)
cmd = [
    sys.executable, "-u", str(work / "phase4_own.py"),
    "--arch", "efficient", "--method", "partial", "--unfreeze", "10",
    "--bpe", str(work / "legal_bpe.json"),
    "--data_dir", str(work / "phase4data"),
    "--weights", str(work / "efficient_model.pt"),
    "--balanced", "--lr", "1e-5", "--head_lr", "1e-4", "--warmup", "80", "--batch", "4",
    "--out", str(work / "{out}"),
] + {extra}
print(" ".join(cmd), flush=True)
code = subprocess.call(cmd, cwd=str(work))
if code != 0:
    raise SystemExit(code)
'''

FAST = ["--focus", "tail", "--cover", "tail", "--chunks", "3", "--chunk_tokens", "256", "--word_cap", "1000", "--epochs", "1", "--max_docs", "12000", "--max_steps", "7000"]
JOBS = [
    ("demonhunter24", "p4-fast-a", FAST + ["--seed", "101"]),
    ("demonhunter24", "p4-fast-b", FAST + ["--seed", "102"]),
    ("anshuman1067", "p4-fast-c", FAST + ["--seed", "103"]),
    ("anshuman1067", "p4-fast-d", FAST + ["--seed", "104"]),
    ("bisendhritlahare", "p4-fast-e", FAST + ["--seed", "105"]),
    ("bisendhritlahare", "p4-fast-f", FAST + ["--seed", "106"]),
]


def main():
    for user, slug, extra in JOBS:
        folder = os.path.join(LOCAL, "kaggle_" + slug.replace("-", "_"))
        os.makedirs(folder, exist_ok=True)
        source = CELL.format(payload=PAYLOAD, words=WORDS_PAYLOAD, out=slug, extra=repr(extra))
        code_file = slug + ".ipynb"
        notebook = {
            "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        with open(os.path.join(folder, code_file), "w", encoding="utf-8") as handle:
            json.dump(notebook, handle)
        metadata = {
            "id": f"{user}/{slug}",
            "title": slug,
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
        print("wrote", folder, flush=True)


if __name__ == "__main__":
    main()
