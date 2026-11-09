"""Download the Phase 5 statute-identification files."""
import os
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Exploration-Lab/CS-781-Capstone"
OUT = r"C:\Users\Anshuman Raj\CS781\phase5data"

names = [name for name in list_repo_files(REPO, repo_type="dataset") if name.startswith("Phase_5")]
print("remote files", names, flush=True)
for name in names:
    path = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset")
    dest = os.path.join(OUT, *name.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.isfile(dest) or os.path.getsize(dest) != os.path.getsize(path):
        shutil.copyfile(path, dest)
    print("saved", dest, os.path.getsize(dest), flush=True)
