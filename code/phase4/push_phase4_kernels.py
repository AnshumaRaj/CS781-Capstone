import json
import os
import subprocess
import time

ACCOUNTS = [
    (
        r"C:\Users\Anshuman Raj\.kaggle\kaggle.json",
        "demonhunter24/phase4-cjpe",
        [
            r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_efficient_lorma",
            r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_hybrid_adapter",
        ],
    ),
    (
        r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle (1).json",
        "bisendhritlahare/phase4-cjpe",
        [
            r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_decoder_bitfit",
            r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_encdec_partial",
        ],
    ),
    (
        r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle_anshu.json",
        "anshuman1067/phase4-cjpe",
        [
            r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_efficient_partial",
            r"C:\Users\Anshuman Raj\CS781\kaggle_phase4_encdec_distill",
        ],
    ),
]


def account_env(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = data["username"]
    env["KAGGLE_KEY"] = data["key"]
    env.pop("KAGGLE_API_TOKEN", None)
    return data["username"], env


def dataset_ready(slug, env):
    result = subprocess.run(
        ["kaggle", "datasets", "status", slug],
        env=env,
        capture_output=True,
        text=True,
    )
    text = (result.stdout + "\n" + result.stderr).strip()
    print(slug, "status_code", result.returncode, flush=True)
    print(text[-500:], flush=True)
    low = text.lower()
    return result.returncode == 0 and ("ready" in low or "complete" in low) and "pending" not in low and "error" not in low


def main():
    for cred, slug, kernels in ACCOUNTS:
        user, env = account_env(cred)
        print("account", user, flush=True)
        ready = False
        for attempt in range(24):
            ready = dataset_ready(slug, env)
            if ready:
                break
            print("waiting", attempt, flush=True)
            time.sleep(30)
        if not ready:
            print("dataset not ready, skip kernels", slug, flush=True)
            continue
        for folder in kernels:
            print("push", os.path.basename(folder), flush=True)
            code = subprocess.call(["kaggle", "kernels", "push", "-p", folder], env=env)
            print("push_code", os.path.basename(folder), code, flush=True)


if __name__ == "__main__":
    main()
