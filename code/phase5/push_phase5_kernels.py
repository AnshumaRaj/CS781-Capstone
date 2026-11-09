"""Push, check, or download the Phase 5 kernels.

    python push_phase5_kernels.py push [slug ...]
    python push_phase5_kernels.py status
    python push_phase5_kernels.py fetch
"""
import json
import os
import subprocess
import sys

from make_phase5_kernels import JOBS, LOCAL

CREDS = {
    "demonhunter24": r"C:\Users\Anshuman Raj\.kaggle\kaggle.json",
    "anshuman1067": r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle_anshu.json",
    "bisendhritlahare": r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle (1).json",
    "shreedhargupta": r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle_iiiii.json",
}
DEST = os.path.join(LOCAL, "phase5_kaggle")


def env_for(user):
    with open(CREDS[user], encoding="utf-8") as handle:
        data = json.load(handle)
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = data["username"]
    env["KAGGLE_KEY"] = data["key"]
    env.pop("KAGGLE_API_TOKEN", None)
    return env


def main():
    action = sys.argv[1]
    only = set(sys.argv[2:])
    for user, slug, _ in JOBS:
        if only and slug not in only:
            continue
        kernel = f"{user}/{slug}"
        if action == "push":
            folder = os.path.join(LOCAL, "kaggle_" + slug.replace("-", "_"))
            code = subprocess.call(["kaggle", "kernels", "push", "-p", folder], env=env_for(user))
            print("push", kernel, code, flush=True)
        elif action == "status":
            result = subprocess.run(["kaggle", "kernels", "status", kernel], env=env_for(user), capture_output=True, text=True)
            print(kernel, (result.stdout + result.stderr).strip().split("status")[-1][:60], flush=True)
        elif action == "fetch":
            out = os.path.join(DEST, slug)
            os.makedirs(out, exist_ok=True)
            subprocess.call(
                ["kaggle", "kernels", "output", kernel, "-p", out, "-o", "-q", "--file-pattern",
                 r"(metrics|thresholds|dev_logits_ids|test_logits_ids)\.json|(dev|test)_logits\.npy|submission\.zip|\.log$"],
                env=env_for(user),
            )
            for root, _, files in os.walk(out):
                if "metrics.json" in files:
                    with open(os.path.join(root, "metrics.json"), encoding="utf-8") as handle:
                        found = json.load(handle)
                    print(slug, os.path.basename(root), "crossval", round(found["f1_per_label_crossval"], 4),
                          "global", round(found["f1_global_cut"], 4), flush=True)


if __name__ == "__main__":
    main()
