import json
import os

from kaggle.api.kaggle_api_extended import KaggleApi

CRED = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\kaggle (1).json"
with open(CRED, encoding="utf-8") as handle:
    data = json.load(handle)
os.environ["KAGGLE_USERNAME"] = data["username"]
os.environ["KAGGLE_KEY"] = data["key"]
os.environ.pop("KAGGLE_API_TOKEN", None)

api = KaggleApi()
api.authenticate()
status = api.kernels_status("bisendhritlahare/phase4-decoder-bitfit")
print(type(status))
print(status)
for name in dir(status):
    if name.startswith("_"):
        continue
    value = getattr(status, name)
    if callable(value):
        continue
    text = str(value)
    if len(text) > 800:
        text = text[:800]
    print(name, "=", text)
