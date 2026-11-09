"""Phase 6 Kaggle kernels: 3 formality runs and 3 score runs.

Formality (spec minimum): decoder LoRMA, efficient adapters, encdec KD.
Score: efficient partial, encdec LoRMA, hybrid partial + hierarchical decode.
"""
import base64
import json
import os
import sys

LOCAL = r"C:\Users\Anshuman Raj\CS781"
ROOT = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781"
FILES = {
    "phase4_own.py": os.path.join(ROOT, "capstone", "phase4", "code", "phase4_own.py"),
    "phase6_summ.py": os.path.join(ROOT, "capstone", "phase6", "code", "phase6_summ.py"),
    "phase1_models.py": os.path.join(ROOT, "capstone", "phase1", "code", "phase1_models.py"),
    "phase2_lner.py": os.path.join(ROOT, "capstone", "phase2", "code", "phase2_lner.py"),
}

LONG = ["--epochs", "12", "--train_minutes", "560", "--batch", "2", "--eval_docs", "16",
        "--gen_tokens", "480", "--min_gen", "180", "--gen_batch", "2", "--require_cuda"]
FIT7H = ["--epochs", "12", "--train_minutes", "360", "--batch", "2", "--eval_docs", "16",
         "--gen_tokens", "480", "--min_gen", "180", "--gen_batch", "2", "--require_cuda"]


def payload(name):
    with open(FILES[name], "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


CELL = r'''import base64, gzip, shutil, subprocess, sys, time
from pathlib import Path
import torch
print("torch", torch.__version__, "cuda_built", torch.version.cuda, flush=True)
print("nvidia-smi", shutil.which("nvidia-smi"), flush=True)
if shutil.which("nvidia-smi"):
    subprocess.call(["nvidia-smi"])
for i in range(90):
    ok = torch.cuda.is_available()
    print("cuda wait", i, ok, flush=True)
    if ok:
        break
    time.sleep(4)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("GPU not attached")
work = Path("/kaggle/working")
(work / "phase4_own.py").write_bytes(base64.b64decode("{own}"))
(work / "phase6_summ.py").write_bytes(base64.b64decode("{summ}"))
(work / "phase1_models.py").write_bytes(base64.b64decode("{p1}"))
(work / "phase2_lner.py").write_bytes(base64.b64decode("{p2}"))
src = sorted(Path("/kaggle/input").rglob("efficient_model.pt"))[0].parent
for name in ("legal_bpe.py", "legal_bpe.json"):
    shutil.copy(src / name, work / name)
data = work / "phase6data" / "Phase_6_Summ"
(data / "Train").mkdir(parents=True, exist_ok=True)
(data / "Test").mkdir(parents=True, exist_ok=True)
print("inputs", [str(p) for p in Path("/kaggle/input").rglob("*") if p.is_file()][:40], flush=True)
for name, dest in (("train.json", data / "Train" / "train.json"), ("test.json", data / "Test" / "test.json")):
    found = sorted(Path("/kaggle/input").rglob("p6_" + name + ".gz")) + sorted(Path("/kaggle/input").rglob("p6_" + name))
    if not found:
        found = sorted(Path("/kaggle/input").rglob("*" + name + ".gz")) + sorted(Path("/kaggle/input").rglob("*" + name))
    if not found:
        raise SystemExit("missing " + name + " under /kaggle/input")
    if found[0].suffix == ".gz":
        with gzip.open(found[0], "rb") as inp, open(dest, "wb") as out:
            shutil.copyfileobj(inp, out)
    else:
        shutil.copy(found[0], dest)
common = ["--bpe", str(work / "legal_bpe.json"), "--data_dir", str(work / "phase6data"), "--require_cuda"]
steps = {steps}
ckpts = sorted(Path("/kaggle/input").rglob("best.pt"))
print("ckpts", ckpts, flush=True)
for out, arch, extra in steps:
    dest = work / out
    dest.mkdir(parents=True, exist_ok=True)
    if ckpts:
        shutil.copy(ckpts[0], dest / "best.pt")
    extra = [item.replace("WORK", str(work)) for item in extra]
    cmd = [sys.executable, "-u", str(work / "phase6_summ.py"), "--arch", arch,
           "--weights", str(src / (arch + "_model.pt")), "--out", str(dest)] + common + extra
    print(" ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(work))
    if code != 0:
        raise SystemExit(code)
'''

JOBS = [
    ("demonhunter24", "p6-enc-score", [
        ("enc_lorma", "encdec", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "8e-5",
                                 "--src_tokens", "1024", "--tgt_tokens", "384"] + LONG),
    ]),
    ("demonhunter24", "p6-eff-partial", [
        ("enc_ext", "encdec", ["--method", "lorma", "--rank", "16", "--alpha", "32",
                               "--src_tokens", "1024", "--decode_only", "--extract_words", "780",
                               "--gen_tokens", "96", "--eval_docs", "12", "--gen_batch", "2", "--require_cuda"]),
    ]),
    ("anshuman1067", "p6-enc-lorma", [
        ("enc_lorma", "encdec", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4",
                                 "--src_tokens", "1024", "--tgt_tokens", "384"] + LONG),
    ]),
    ("anshuman1067", "p6-eff-hier", [
        ("enc_ext", "encdec", ["--method", "lorma", "--rank", "16", "--alpha", "32",
                               "--src_tokens", "1024", "--decode_only", "--extract_words", "820",
                               "--gen_tokens", "96", "--eval_docs", "12", "--gen_batch", "2", "--require_cuda"]),
    ]),
    ("bisendhritlahare", "p6-enc-kd", [
        ("eff_teacher", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4",
                                      "--src_tokens", "624", "--tgt_tokens", "384", "--epochs", "6",
                                      "--train_minutes", "240", "--batch", "2", "--eval_docs", "16",
                                      "--gen_tokens", "400", "--min_gen", "160", "--gen_batch", "2"]),
        ("enc_kd", "encdec", ["--method", "partial", "--unfreeze", "6", "--lr", "2e-5",
                              "--src_tokens", "768", "--tgt_tokens", "320",
                              "--teacher_weights", "WORK/eff_teacher/best.pt",
                              "--teacher_arch", "efficient", "--teacher_method", "lorma",
                              "--kd_weight", "0.5", "--kd_temp", "2.0",
                              "--epochs", "8", "--train_minutes", "300", "--batch", "2",
                              "--eval_docs", "16", "--gen_tokens", "480", "--min_gen", "180",
                              "--gen_batch", "2"]),
    ]),
    ("bisendhritlahare", "p6-eff-score", [
        ("enc_ext", "encdec", ["--method", "partial", "--unfreeze", "6",
                               "--src_tokens", "768", "--decode_only", "--extract_words", "640",
                               "--gen_tokens", "120", "--eval_docs", "16", "--gen_batch", "2", "--require_cuda"]),
    ]),
]


def main():
    only = set(sys.argv[1:])
    own, summ, p1, p2 = payload("phase4_own.py"), payload("phase6_summ.py"), payload("phase1_models.py"), payload("phase2_lner.py")
    kernel_src = {
        "p6-eff-partial": ["demonhunter24/p6-enc-score"],
        "p6-eff-hier": ["anshuman1067/p6-enc-lorma"],
        "p6-eff-score": ["bisendhritlahare/p6-enc-kd"],
    }
    for user, slug, steps in JOBS:
        if only and slug not in only:
            continue
        folder = os.path.join(LOCAL, "kaggle_" + slug.replace("-", "_"))
        os.makedirs(folder, exist_ok=True)
        source = CELL.format(own=own, summ=summ, p1=p1, p2=p2, steps=repr(steps))
        notebook = {
            "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        with open(os.path.join(folder, slug + ".ipynb"), "w", encoding="utf-8") as handle:
            json.dump(notebook, handle)
        metadata = {
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": slug + ".ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "false",
            "dataset_sources": [f"{user}/phase4-cjpe", f"{user}/phase6-summ"],
            "kernel_sources": kernel_src.get(slug, []),
            "competition_sources": [],
            "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
        }
        with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print("wrote", folder, flush=True)


if __name__ == "__main__":
    main()
