"""Private phase7-pcr dataset on each Kaggle GPU account: gzipped split JSON."""
import gzip
import json
import os
import shutil
import sys

import requests
import urllib3

urllib3.disable_warnings()
_send = requests.Session.send


def _send_unverified(self, request, **kwargs):
    kwargs["verify"] = False
    return _send(self, request, **kwargs)


requests.Session.send = _send_unverified

from kaggle.api.kaggle_api_extended import KaggleApi

sys.path.insert(0, r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase5\code")
from push_phase5_kernels import CREDS, env_for

LOCAL = r"C:\Users\Anshuman Raj\CS781"
DATA = os.path.join(LOCAL, r"phase7data\Phase_7_PCR")
PACKED = os.path.join(LOCAL, "phase7_packed")
USERS = [user for user in CREDS if user != "shreedhargupta"]
FILES = (
    ("Train", "train.json"),
    ("Train", "train_candidates.json"),
    ("Dev", "dev.json"),
    ("Dev", "dev_candidates.json"),
    ("Test", "test.json"),
    ("Test", "test_candidates.json"),
)


def pack():
    os.makedirs(PACKED, exist_ok=True)
    for split, name in FILES:
        dest = os.path.join(PACKED, f"p7_{name}.gz")
        if not os.path.isfile(dest) or os.path.getsize(dest) < 1000:
            with open(os.path.join(DATA, split, name), "rb") as inp, gzip.open(dest, "wb", compresslevel=6) as out:
                shutil.copyfileobj(inp, out)
        print("packed", dest, os.path.getsize(dest) // 1_000_000, "MB", flush=True)


def main():
    pack()
    only = set(sys.argv[1:])
    for user in USERS:
        if only and user not in only:
            continue
        folder = os.path.join(LOCAL, f"kaggle_phase7_data_{user}")
        os.makedirs(folder, exist_ok=True)
        for name in os.listdir(PACKED):
            shutil.copyfile(os.path.join(PACKED, name), os.path.join(folder, name))
        with open(os.path.join(folder, "dataset-metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({"title": "phase7-pcr", "id": f"{user}/phase7-pcr", "licenses": [{"name": "CC0-1.0"}]}, handle)
        os.environ.pop("KAGGLE_API_TOKEN", None)
        for key, value in env_for(user).items():
            if key.startswith("KAGGLE"):
                os.environ[key] = value
        api = KaggleApi()
        api.authenticate()
        try:
            api.dataset_create_new(folder, public=False, quiet=True, dir_mode="zip")
            print("create", user, "ok", flush=True)
        except Exception as error:
            message = str(error)
            if "already exists" in message.lower() or "409" in message:
                api.dataset_create_version(folder, version_notes="phase7 train/dev/test", quiet=True, dir_mode="zip")
                print("version", user, "ok", flush=True)
            else:
                print("fail", user, message[:400], flush=True)


if __name__ == "__main__":
    main()
