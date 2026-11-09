"""Kaggle-only: quota, kernel status, fetch completed Phase 6 zips. No local train."""
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()
_send = requests.Session.send


def _send_unverified(self, request, **kwargs):
    kwargs["verify"] = False
    return _send(self, request, **kwargs)


requests.Session.send = _send_unverified

ROOT = Path(r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781")
sys.path.insert(0, str(ROOT / "capstone" / "phase5" / "code"))
sys.path.insert(0, str(ROOT / "capstone" / "phase6" / "code"))

from kaggle.api.kaggle_api_extended import KaggleApi
from make_phase6_kernels import JOBS
from push_phase5_kernels import CREDS, env_for

OUT = Path(r"C:\Users\Anshuman Raj\CS781")
SUB = ROOT / "capstone" / "phase6" / "submissions"
PATTERN = r"(metrics\.json|submission\.zip|prediction\.json)"
NAMES = {
    "p6-enc-score": "encdec_lorma_kaggle",
    "p6-enc-lorma": "encdec_lorma_kaggle2",
    "p6-eff-partial": "encdec_extract_v5_780",
    "p6-eff-hier": "encdec_extract_v5_820",
    "p6-enc-kd": "encdec_kd_kaggle",
    "p6-eff-score": "encdec_extract_640",
}


def api_for(user):
    os.environ.pop("KAGGLE_API_TOKEN", None)
    for key, value in env_for(user).items():
        if key.startswith("KAGGLE_"):
            os.environ[key] = value
    api = KaggleApi()
    api.authenticate()
    return api


print("=== GPU quota ===", flush=True)
for user in CREDS:
    api = api_for(user)
    response = api.quota_view()
    gpu = response.gpu_quota
    used = gpu.time_used.total_seconds() / 3600
    total = gpu.total_time_allowed.total_seconds() / 3600
    print(f"{user}: {used:.2f}/{total:.2f}h remaining {max(0.0, total - used):.2f}h", flush=True)

print("\n=== kernels ===", flush=True)
done = []
for user, slug, _ in JOBS:
    api = api_for(user)
    kernel = f"{user}/{slug}"
    try:
        status = api.kernels_status(kernel)
        fail = getattr(status, "failure_message", "") or ""
        print(kernel, status.status, fail, flush=True)
        if str(status.status).endswith("COMPLETE"):
            done.append((user, slug))
    except Exception as error:
        print(kernel, type(error).__name__, str(error)[:180], flush=True)

print("\n=== fetch COMPLETE ===", flush=True)
SUB.mkdir(parents=True, exist_ok=True)
for user, slug in done:
    api = api_for(user)
    dest = OUT / ("kaggle_out_" + slug.replace("-", "_") + "_slim")
    dest.mkdir(parents=True, exist_ok=True)
    print("fetch", f"{user}/{slug}", flush=True)
    try:
        api.kernels_output(f"{user}/{slug}", path=str(dest), file_pattern=PATTERN, force=True, quiet=True)
    except Exception as error:
        print("  output", type(error).__name__, str(error)[:240], flush=True)
        continue
    combo = None
    zip_path = None
    for root, _, files in os.walk(dest):
        if "metrics.json" in files:
            report = json.load(open(os.path.join(root, "metrics.json"), encoding="utf-8"))
            print("  metrics", report, flush=True)
            combo = float(report.get("combo", 0.0))
        if "submission.zip" in files:
            zip_path = os.path.join(root, "submission.zip")
    if zip_path and combo is not None:
        tag = f"{combo:.4f}".split(".")[1]
        name = NAMES.get(slug, slug.replace("-", "_"))
        folder = SUB / f"{name}-score(.{tag})"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, folder / "submission.zip")
        print("  packed", folder, zipfile.ZipFile(zip_path).namelist(), flush=True)
