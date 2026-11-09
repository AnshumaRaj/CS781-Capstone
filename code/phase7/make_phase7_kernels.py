"""Phase 7 Kaggle kernels: TF-IDF plus four Phase 1 dual-encoder blends."""
import base64
import json
import os
import sys

LOCAL = r"C:\Users\Anshuman Raj\CS781"
ROOT = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781"
FILES = {
    "phase4_own.py": os.path.join(ROOT, "capstone", "phase4", "code", "phase4_own.py"),
    "phase7_pcr.py": os.path.join(ROOT, "capstone", "phase7", "code", "phase7_pcr.py"),
    "phase1_models.py": os.path.join(ROOT, "capstone", "phase1", "code", "phase1_models.py"),
    "phase2_lner.py": os.path.join(ROOT, "capstone", "phase2", "code", "phase2_lner.py"),
}

FIT = ["--epochs", "8", "--train_minutes", "360", "--batch", "2", "--negs", "4",
       "--windows", "4", "--width", "256", "--pool_batch", "8", "--require_cuda"]
HARD = ["--mode", "blend", "--hard_negs", "--rerank_top", "50", "--rerank_mix", "0.4",
        "--word_cap", "8000", "--ngram", "3", "--features", "400000",
        "--epochs", "8", "--train_minutes", "90", "--batch", "2", "--negs", "6",
        "--windows", "4", "--width", "256", "--pool_batch", "8", "--require_cuda"]
PAIR = ["--rerank_top", "20",
        "--word_cap", "8000", "--ngram", "3", "--features", "400000",
        "--epochs", "3", "--train_minutes", "32", "--batch", "4", "--negs", "8",
        "--qlen", "256", "--dlen", "256", "--require_cuda"]
KD_TEACH = ["--mode", "blend", "--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4",
            "--epochs", "6", "--train_minutes", "180", "--batch", "2", "--negs", "4",
            "--windows", "4", "--width", "256", "--pool_batch", "8", "--require_cuda"]
KD_STUD = ["--mode", "blend", "--method", "partial", "--unfreeze", "6", "--lr", "2e-5",
           "--epochs", "8", "--train_minutes", "240", "--batch", "2", "--negs", "4",
           "--windows", "4", "--width", "256", "--pool_batch", "8",
           "--teacher_base", "WORKSRC/efficient_model.pt",
           "--teacher_weights", "WORK/eff_teacher/best.pt", "--teacher_arch", "efficient",
           "--teacher_method", "lorma", "--kd_weight", "0.5", "--kd_temp", "2.0", "--require_cuda"]


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
need_gpu = {need_gpu}
if need_gpu:
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
(work / "phase7_pcr.py").write_bytes(base64.b64decode("{pcr}"))
(work / "phase1_models.py").write_bytes(base64.b64decode("{p1}"))
(work / "phase2_lner.py").write_bytes(base64.b64decode("{p2}"))
src = sorted(Path("/kaggle/input").rglob("efficient_model.pt"))[0].parent
for name in ("legal_bpe.py", "legal_bpe.json"):
    shutil.copy(src / name, work / name)
data = work / "phase7data" / "Phase_7_PCR"
for split in ("Train", "Dev", "Test"):
    (data / split).mkdir(parents=True, exist_ok=True)
print("inputs", [str(p) for p in Path("/kaggle/input").rglob("*") if p.is_file()][:80], flush=True)
mapping = {{
    "p7_train.json.gz": data / "Train" / "train.json",
    "p7_train_candidates.json.gz": data / "Train" / "train_candidates.json",
    "p7_dev.json.gz": data / "Dev" / "dev.json",
    "p7_dev_candidates.json.gz": data / "Dev" / "dev_candidates.json",
    "p7_test.json.gz": data / "Test" / "test.json",
    "p7_test_candidates.json.gz": data / "Test" / "test_candidates.json",
}}
for name, dest in mapping.items():
    stem = name[:-3] if name.endswith(".gz") else name
    found = sorted(Path("/kaggle/input").rglob(name)) + sorted(Path("/kaggle/input").rglob(stem))
    if not found:
        raise SystemExit("missing " + name)
    if found[0].suffix == ".gz":
        with gzip.open(found[0], "rb") as inp, open(dest, "wb") as out:
            shutil.copyfileobj(inp, out)
    else:
        shutil.copy(found[0], dest)
    print("unpacked", name, dest.stat().st_size, flush=True)
common = ["--bpe", str(work / "legal_bpe.json"), "--data_dir", str(work / "phase7data")]
steps = {steps}
for out, arch, extra in steps:
    dest = work / out
    dest.mkdir(parents=True, exist_ok=True)
    extra = [item.replace("WORKSRC", str(src)).replace("WORK", str(work)) for item in extra]
    cmd = [sys.executable, "-u", str(work / "phase7_pcr.py"), "--arch", arch,
           "--weights", str(src / (arch + "_model.pt")), "--out", str(dest)] + common + extra
    print(" ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(work))
    if code != 0:
        raise SystemExit(code)
'''

JOBS = [
    ("demonhunter24", "p7-eff-lorma", True, [
        ("eff_lorma", "efficient", ["--mode", "blend", "--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4"] + FIT),
    ]),
    ("demonhunter24", "p7-tfidf", False, [
        ("tfidf", "efficient", ["--mode", "tfidf"]),
    ]),
    ("anshuman1067", "p7-dec-lorma", True, [
        ("dec_lorma", "decoder", ["--mode", "blend", "--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4"] + FIT),
    ]),
    ("anshuman1067", "p7-eff-partial", True, [
        ("eff_partial", "efficient", ["--mode", "blend", "--method", "partial", "--unfreeze", "6", "--lr", "2e-5"] + FIT),
    ]),
    ("bisendhritlahare", "p7-hyb-adapter", True, [
        ("hyb_adapter", "hybrid", ["--mode", "blend", "--method", "adapter", "--rank", "16", "--lr", "1e-4"] + FIT),
    ]),
    ("bisendhritlahare", "p7-enc-kd", True, [
        ("eff_teacher", "efficient", KD_TEACH),
        ("enc_kd", "encdec", KD_STUD),
    ]),
    ("demonhunter24", "p7-hard-lorma", True, [
        ("hard_lorma", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4"] + HARD),
    ]),
    ("anshuman1067", "p7-hard-partial", True, [
        ("hard_partial", "efficient", ["--method", "partial", "--unfreeze", "6", "--lr", "2e-5"] + HARD),
    ]),
    ("bisendhritlahare", "p7-hard-enc", True, [
        ("hard_enc", "encdec", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4"] + HARD),
    ]),
    ("demonhunter24", "p7-pair-eff", True, [
        ("pair_eff", "efficient", ["--mode", "pair", "--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4"] + PAIR),
    ]),
    ("bisendhritlahare", "p7-pair-enc", True, [
        ("pair_enc", "encdec", ["--mode", "pair", "--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4"] + PAIR),
    ]),
    ("anshuman1067", "p7-pair-dec", True, [
        ("pair_dec", "decoder", ["--mode", "pair", "--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "1e-4"] + PAIR),
    ]),
]


def main():
    only = set(sys.argv[1:])
    own, pcr, p1, p2 = payload("phase4_own.py"), payload("phase7_pcr.py"), payload("phase1_models.py"), payload("phase2_lner.py")
    for user, slug, need_gpu, steps in JOBS:
        if only and slug not in only:
            continue
        folder = os.path.join(LOCAL, "kaggle_" + slug.replace("-", "_"))
        os.makedirs(folder, exist_ok=True)
        source = CELL.format(own=own, pcr=pcr, p1=p1, p2=p2, steps=repr(steps), need_gpu=repr(need_gpu))
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
            "enable_gpu": "true" if need_gpu else "false",
            "enable_tpu": "false",
            "enable_internet": "false",
            "dataset_sources": [f"{user}/phase4-cjpe", f"{user}/phase7-pcr"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        if need_gpu:
            metadata["machine_shape"] = "NvidiaTeslaT4"
        with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print("wrote", folder, flush=True)


if __name__ == "__main__":
    main()
