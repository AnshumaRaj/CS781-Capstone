"""Poll Phase 5 kernels and download each one when it finishes.

The laptop clock is behind, so Kaggle's certificate looks not-yet-valid.
TLS verification is turned off for these calls only.

    python -u wait_phase5.py [hours] [slug ...]
"""
import json
import os
import sys
import time

import requests
import urllib3

urllib3.disable_warnings()
_send = requests.Session.send


def _send_unverified(self, request, **kwargs):
    kwargs["verify"] = False
    return _send(self, request, **kwargs)


requests.Session.send = _send_unverified

from kaggle.api.kaggle_api_extended import KaggleApi

from make_phase5_kernels import JOBS
from push_phase5_kernels import DEST, env_for

DONE = ("COMPLETE", "ERROR", "CANCEL")
PATTERN = r"(metrics|thresholds|dev_logits_ids|test_logits_ids)\.json|(dev|test)_logits\.npy|submission\.zip|\.log$"


def api_for(user):
    os.environ.pop("KAGGLE_API_TOKEN", None)
    for key, value in env_for(user).items():
        if key.startswith("KAGGLE_"):
            os.environ[key] = value
    api = KaggleApi()
    api.authenticate()
    return api


def kernel_state(user, slug):
    response = api_for(user).kernels_status(f"{user}/{slug}")
    return str(response.status)


def fetch(user, slug):
    out = os.path.join(DEST, slug)
    os.makedirs(out, exist_ok=True)
    api_for(user).kernels_output(f"{user}/{slug}", path=out, file_pattern=PATTERN, force=True, quiet=True)
    for root, _, files in os.walk(out):
        if "metrics.json" in files:
            with open(os.path.join(root, "metrics.json"), encoding="utf-8") as handle:
                report = json.load(handle)
            print(slug, os.path.basename(root),
                  "crossval", round(report.get("f1_per_label_crossval", -1), 4),
                  "global", round(report.get("f1_global_cut", -1), 4), flush=True)


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][:1].isdigit() else 10
    only = {arg for arg in sys.argv[1:] if not arg[:1].isdigit()}
    limit = time.time() + hours * 3600
    pending = {slug: user for user, slug, _ in JOBS if not only or slug in only}
    while pending and time.time() < limit:
        for slug, user in list(pending.items()):
            state = kernel_state(user, slug)
            if any(word in state.upper() for word in DONE):
                print(time.strftime("%H:%M"), slug, state, flush=True)
                fetch(user, slug)
                pending.pop(slug)
        if pending:
            print(time.strftime("%H:%M"), "waiting on", ", ".join(sorted(pending)), flush=True)
            time.sleep(600)
    print("finished; still pending:", sorted(pending), flush=True)


if __name__ == "__main__":
    main()
