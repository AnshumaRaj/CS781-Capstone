"""Download the Phase 3 rhetorical-role files into a local folder."""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

REPO = "Exploration-Lab/CS-781-Capstone"
OUT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Anshuman Raj\CS781\phase3data"
FILES = [
    "Phase_3_RR/Train/label_vocab.json",
    "Phase_3_RR/Train/CL_train.json",
    "Phase_3_RR/Train/IT_train.json",
    "Phase_3_RR/Train/CL_dev.json",
    "Phase_3_RR/Train/IT_dev.json",
    "Phase_3_RR/Test/CL_test.json",
    "Phase_3_RR/Test/IT_test.json",
]

for name in FILES:
    path = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset")
    dest = os.path.join(OUT, *name.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(path, dest)
    print("saved", dest, os.path.getsize(dest), flush=True)
