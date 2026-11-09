"""Phase 5 Kaggle kernels. Each notebook carries the current trainers and runs a list of steps."""
import base64
import json
import os
import sys

LOCAL = r"C:\Users\Anshuman Raj\CS781"
CODE = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781"


def payload(name):
    with open(os.path.join(CODE, name), "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


CELL = r'''import base64, gzip, shutil, subprocess, sys
from pathlib import Path
work = Path("/kaggle/working")
(work / "phase4_own.py").write_bytes(base64.b64decode("{own}"))
(work / "phase5_lsi.py").write_bytes(base64.b64decode("{lsi}"))
src = sorted(Path("/kaggle/input").rglob("efficient_model.pt"))[0].parent
for name in ("phase1_models.py", "phase2_lner.py", "legal_bpe.py", "legal_bpe.json"):
    shutil.copy(src / name, work / name)
data = work / "phase5data" / "Phase_5_LSI"
(data / "Train").mkdir(parents=True, exist_ok=True)
(data / "Test").mkdir(parents=True, exist_ok=True)
for name, dest in (("train.json", data / "Train" / "train.json"), ("dev.json", data / "Train" / "dev.json"), ("test.json", data / "Test" / "test.json")):
    found = sorted(Path("/kaggle/input").rglob("p5_" + name + ".gz")) + sorted(Path("/kaggle/input").rglob("p5_" + name))
    if not found:
        raise SystemExit("missing " + name)
    if found[0].suffix == ".gz":
        with gzip.open(found[0], "rb") as inp, open(dest, "wb") as out:
            shutil.copyfileobj(inp, out)
    else:
        shutil.copy(found[0], dest)
common = ["--bpe", str(work / "legal_bpe.json"), "--data_dir", str(work / "phase5data"), "--cache_dir", str(work / "cache")]
steps = {steps}
marker = sorted(Path("/kaggle/input").rglob("tfidf_base.txt"))
base = str(marker[0].parent) if marker else "missing-tfidf-base"
for out, arch, extra in steps:
    extra = [item.replace("WORK", str(work)).replace("BASE", base) for item in extra]
    cmd = [sys.executable, "-u", str(work / "phase5_lsi.py"), "--arch", arch, "--weights", str(src / (arch + "_model.pt")), "--out", str(work / out)] + common + extra
    print(" ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(work))
    if code != 0:
        raise SystemExit(code)
'''

FULL = ["--epochs", "2", "--eval_every", "3000"]
JOBS = [
    ("demonhunter24", "p5-eff-lorma", [
        ("eff_lorma", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4"] + FULL),
    ]),
    ("demonhunter24", "p5-eff-kd", [
        ("eff_partial", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "2e-5", "--dump_train"] + FULL),
        ("enc_kd", "encdec", ["--method", "partial", "--unfreeze", "6", "--lr", "3e-5",
                              "--teacher_logits", "WORK/eff_partial/train_logits.npy"] + FULL),
    ]),
    ("anshuman1067", "p5-dec-partial", [
        ("dec_partial", "decoder", ["--method", "partial", "--unfreeze", "8", "--lr", "2e-5"] + FULL),
    ]),
    ("anshuman1067", "p5-enc-partial", [
        ("enc_partial", "encdec", ["--method", "partial", "--unfreeze", "6", "--lr", "3e-5"] + FULL),
    ]),
    ("bisendhritlahare", "p5-hyb-adapter", [
        ("hyb_adapter", "hybrid", ["--method", "adapter", "--rank", "64", "--lr", "5e-4"] + FULL),
    ]),
    ("bisendhritlahare", "p5-dec-lorma", [
        ("dec_lorma", "decoder", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4"] + FULL),
    ]),
    ("anshuman1067", "p5-enc-hi", [
        ("enc_hi", "encdec", ["--method", "partial", "--unfreeze", "6", "--lr", "1e-4", "--pos_weight", "0.5",
                              "--epochs", "3", "--eval_every", "3000"]),
    ]),
    ("demonhunter24", "p5-eff-long", [
        ("eff_long", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4",
                                   "--pos_weight", "0.5", "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("anshuman1067", "p5-eff-r64", [
        ("eff_r64", "efficient", ["--method", "lorma", "--rank", "64", "--alpha", "64", "--lr", "3e-4",
                                  "--pos_weight", "0.5", "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("anshuman1067", "p5-eff-res", [
        ("eff_res", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4",
                                  "--base_logits", "BASE", "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-dec-hi", [
        ("dec_hi", "decoder", ["--method", "partial", "--unfreeze", "8", "--lr", "5e-5", "--pos_weight", "0.5",
                               "--epochs", "3", "--eval_every", "3000"]),
    ]),
    ("bisendhritlahare", "p5-hyb-hi", [
        ("hyb_hi", "hybrid", ["--method", "partial", "--unfreeze", "8", "--lr", "5e-5", "--pos_weight", "0.5",
                              "--epochs", "3", "--eval_every", "3000"]),
    ]),
    ("demonhunter24", "p5-eff-part4", [
        ("eff_part4", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                    "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("demonhunter24", "p5-eff-ctx", [
        ("eff_ctx", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4",
                                  "--pos_weight", "0.5", "--chunks", "10", "--word_cap", "4000",
                                  "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-eff-asl", [
        ("eff_asl", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4",
                                  "--loss", "asl", "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("demonhunter24", "p5-eff-partctx", [
        ("eff_partctx", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                      "--chunks", "8", "--word_cap", "4000", "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-eff-s2", [
        ("eff_s2", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4",
                                 "--pos_weight", "0.5", "--seed", "7", "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("demonhunter24", "p5-eff-part10", [
        ("eff_part10", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                     "--chunks", "10", "--word_cap", "4000", "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-eff-parts2", [
        ("eff_parts2", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                     "--seed", "7", "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("anshuman1067", "p5-eff-parthi", [
        ("eff_parthi", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "5e-5", "--pos_weight", "0.5",
                                     "--seed", "11", "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("anshuman1067", "p5-eff-ctx2", [
        ("eff_ctx2", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4",
                                   "--pos_weight", "0.5", "--chunks", "10", "--word_cap", "4000", "--seed", "7",
                                   "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-eff-part10b", [
        ("eff_part10b", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                      "--chunks", "10", "--word_cap", "4000", "--seed", "13",
                                      "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-eff-w384", [
        ("eff_w384", "efficient", ["--method", "lorma", "--rank", "16", "--alpha", "32", "--lr", "2e-4",
                                   "--pos_weight", "0.5", "--chunk_tokens", "384", "--word_cap", "4000",
                                   "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("demonhunter24", "p5-eff-hier", [
        ("eff_hier", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                   "--hier", "--chunks", "10", "--word_cap", "4000", "--epochs", "4",
                                   "--eval_every", "6000"]),
    ]),
    ("demonhunter24", "p5-eff-longctx", [
        ("eff_longctx", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                      "--chunks", "12", "--chunk_tokens", "256", "--word_cap", "5000",
                                      "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("anshuman1067", "p5-eff-traindev", [
        ("eff_traindev", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                       "--include_dev", "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("anshuman1067", "p5-eff-aslft", [
        ("eff_aslft", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--loss", "asl",
                                    "--epochs", "4", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-eff-hierdev", [
        ("eff_hierdev", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                      "--hier", "--include_dev", "--chunks", "8", "--word_cap", "4000",
                                      "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("demonhunter24", "p5-eff-final", [
        ("eff_final", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                    "--include_dev", "--hier", "--chunks", "10", "--word_cap", "4000",
                                    "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("anshuman1067", "p5-eff-final2", [
        ("eff_final2", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                     "--include_dev", "--chunk_tokens", "384", "--word_cap", "4000",
                                     "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-eff-final3", [
        ("eff_final3", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                     "--include_dev", "--hier", "--seed", "21", "--epochs", "3", "--eval_every", "6000"]),
    ]),
    ("bisendhritlahare", "p5-eff-w384ft", [
        ("eff_w384ft", "efficient", ["--method", "partial", "--unfreeze", "10", "--lr", "3e-5", "--pos_weight", "0.5",
                                     "--chunk_tokens", "384", "--word_cap", "4000", "--epochs", "3",
                                     "--eval_every", "6000"]),
    ]),
]


def main():
    only = set(sys.argv[1:])
    own, lsi = payload("phase4_own.py"), payload("phase5_lsi.py")
    for user, slug, steps in JOBS:
        if only and slug not in only:
            continue
        folder = os.path.join(LOCAL, "kaggle_" + slug.replace("-", "_"))
        os.makedirs(folder, exist_ok=True)
        source = CELL.format(own=own, lsi=lsi, steps=repr(steps))
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
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [f"{user}/phase4-cjpe", f"{user}/phase5-lsi"]
            + ([f"{user}/phase5-tfidf"] if "BASE" in repr(steps) else []),
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
