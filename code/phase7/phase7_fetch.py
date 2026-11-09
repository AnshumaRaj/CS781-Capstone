"""Download Phase 7 Prior Case Retrieval files. TLS off (laptop clock)."""
import os
import shutil

import requests
import urllib3

urllib3.disable_warnings()
_send = requests.Session.send


def _send_unverified(self, request, **kwargs):
    kwargs["verify"] = False
    return _send(self, request, **kwargs)


requests.Session.send = _send_unverified

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Exploration-Lab/CS-781-Capstone"
OUT = r"C:\Users\Anshuman Raj\CS781\phase7data"

names = [name for name in list_repo_files(REPO, repo_type="dataset") if name.startswith("Phase_7")]
print("remote files", names, flush=True)
readme = hf_hub_download(repo_id=REPO, filename="README.md", repo_type="dataset")
print("README", readme, flush=True)
print(open(readme, encoding="utf-8", errors="replace").read()[:12000], flush=True)
for name in names:
    path = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset")
    dest = os.path.join(OUT, *name.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.isfile(dest) or os.path.getsize(dest) != os.path.getsize(path):
        shutil.copyfile(path, dest)
    print("saved", dest, os.path.getsize(dest), flush=True)
