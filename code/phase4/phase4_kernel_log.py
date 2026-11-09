import json
import os

import requests

CRED = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle (1).json"
with open(CRED, encoding="utf-8") as handle:
    data = json.load(handle)

url = "https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionLogsStream"
body = {
    "userName": "bisendhritlahare",
    "kernelSlug": "phase4-decoder-bitfit",
    "waitForLogsUrlSeconds": 5,
}
response = requests.post(url, json=body, auth=(data["username"], data["key"]), timeout=60)
print("status", response.status_code)
print("ctype", response.headers.get("Content-Type", ""))
text = response.text
out = r"C:\Users\Anshuman Raj\CS781\phase4_kaggle_logs\decoder_bitfit.log"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as handle:
    handle.write(text)
print("wrote", out, "len", len(text))
