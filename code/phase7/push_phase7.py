"""Push Phase 7 kernels. TLS verification is off because the laptop clock is behind."""
import os
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
sys.path.insert(0, r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone\phase7\code")

from make_phase7_kernels import JOBS, LOCAL
from push_phase5_kernels import env_for


def main(wanted=None):
    wanted = set(wanted if wanted is not None else sys.argv[1:])
    for user, slug, need_gpu, _ in JOBS:
        if wanted and slug not in wanted:
            continue
        os.environ.pop("KAGGLE_API_TOKEN", None)
        for key, value in env_for(user).items():
            if key.startswith("KAGGLE"):
                os.environ[key] = value
        api = KaggleApi()
        api.authenticate()
        folder = os.path.join(LOCAL, "kaggle_" + slug.replace("-", "_"))
        kwargs = {"acc": "NvidiaTeslaT4"} if need_gpu else {}
        response = api.kernels_push(folder, **kwargs)
        print(
            "push",
            f"{user}/{slug}",
            getattr(response, "ref", ""),
            "v",
            getattr(response, "version_number", None),
            getattr(response, "error", "") or "ok",
            flush=True,
        )


if __name__ == "__main__":
    main()
