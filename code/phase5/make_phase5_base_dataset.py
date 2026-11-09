"""Create a private phase5-tfidf Kaggle dataset (out-of-fold TF-IDF logits) on the given accounts.

    python make_phase5_base_dataset.py RUN_DIR user [user ...]
"""
import json
import os
import shutil
import subprocess
import sys

from make_phase5_dataset import LOCAL, env_for

FILES = [f"{split}_logits{ext}" for split in ("train", "dev", "test") for ext in (".npy", "_ids.json")]


def main():
    run, users = sys.argv[1], sys.argv[2:]
    for user in users:
        folder = os.path.join(LOCAL, f"kaggle_phase5_tfidf_{user}")
        os.makedirs(folder, exist_ok=True)
        for name in FILES:
            shutil.copyfile(os.path.join(run, name), os.path.join(folder, name))
        with open(os.path.join(folder, "tfidf_base.txt"), "w", encoding="utf-8") as handle:
            handle.write("out-of-fold TF-IDF logistic regression logits\n")
        with open(os.path.join(folder, "dataset-metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({"title": "phase5-tfidf", "id": f"{user}/phase5-tfidf", "licenses": [{"name": "CC0-1.0"}]}, handle)
        code = subprocess.call(["kaggle", "datasets", "create", "-p", folder, "-q"], env=env_for(user))
        print("create", user, code, flush=True)


if __name__ == "__main__":
    main()
