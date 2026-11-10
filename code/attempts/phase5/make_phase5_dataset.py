"""Create a private phase5-lsi Kaggle dataset on each account: gzipped data plus the Phase 5 code."""
import gzip
import json
import os
import shutil
import subprocess

LOCAL = r"C:\Users\Anshuman Raj\CS781"
DATA = os.path.join(LOCAL, r"phase5data\Phase_5_LSI")
CODE = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781"
PACKED = os.path.join(LOCAL, "phase5_packed")
CREDS = {
    "demonhunter24": r"C:\Users\Anshuman Raj\.kaggle\kaggle.json",
    "anshuman1067": r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle_anshu.json",
    "bisendhritlahare": r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle (1).json",
}


def env_for(user):
    with open(CREDS[user], encoding="utf-8") as handle:
        data = json.load(handle)
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = data["username"]
    env["KAGGLE_KEY"] = data["key"]
    env.pop("KAGGLE_API_TOKEN", None)
    return env


def pack():
    os.makedirs(PACKED, exist_ok=True)
    for split, name in (("Train", "train.json"), ("Train", "dev.json"), ("Test", "test.json")):
        dest = os.path.join(PACKED, f"p5_{name}.gz")
        if not os.path.isfile(dest):
            with open(os.path.join(DATA, split, name), "rb") as inp, gzip.open(dest, "wb", compresslevel=6) as out:
                shutil.copyfileobj(inp, out)
        print("packed", dest, os.path.getsize(dest) // 1_000_000, "MB", flush=True)


def main():
    pack()
    for user in CREDS:
        folder = os.path.join(LOCAL, f"kaggle_phase5_data_{user}")
        os.makedirs(folder, exist_ok=True)
        for name in os.listdir(PACKED):
            target = os.path.join(folder, name)
            if not os.path.isfile(target):
                shutil.copyfile(os.path.join(PACKED, name), target)
        with open(os.path.join(folder, "dataset-metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({"title": "phase5-lsi", "id": f"{user}/phase5-lsi", "licenses": [{"name": "CC0-1.0"}]}, handle)
        code = subprocess.call(["kaggle", "datasets", "create", "-p", folder, "-q"], env=env_for(user))
        print("create", user, code, flush=True)


if __name__ == "__main__":
    main()
