"""Download the Phase 4 court-judgment files."""
import os
import shutil

from huggingface_hub import hf_hub_download

REPO = "Exploration-Lab/CS-781-Capstone"
OUT = r"C:\Users\Anshuman Raj\CS781\phase4data"
FILES = [
    "Phase_4_CJPE/Train/train.json",
    "Phase_4_CJPE/Train/dev.json",
    "Phase_4_CJPE/Test/test.json",
]

for name in FILES:
    path = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset")
    dest = os.path.join(OUT, *name.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.isfile(dest) or os.path.getsize(dest) != os.path.getsize(path):
        shutil.copyfile(path, dest)
    print("saved", dest, os.path.getsize(dest), flush=True)
