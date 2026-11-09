import json
import os
import subprocess

JOBS = [
    (r"C:\Users\Anshuman Raj\.kaggle\kaggle.json", [
        r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_efficient_lorma_long",
        r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_efficient_lorma_s7",
    ]),
    (r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle_anshu.json", [
        r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_efficient_partial_long",
        r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_efficient_partial_s7",
    ]),
]

for cred, folders in JOBS:
    with open(cred, encoding="utf-8") as handle:
        data = json.load(handle)
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = data["username"]
    env["KAGGLE_KEY"] = data["key"]
    env.pop("KAGGLE_API_TOKEN", None)
    print("account", data["username"], flush=True)
    for folder in folders:
        print("push", os.path.basename(folder), flush=True)
        code = subprocess.call(["kaggle", "kernels", "push", "-p", folder], env=env)
        print("push_code", os.path.basename(folder), code, flush=True)
