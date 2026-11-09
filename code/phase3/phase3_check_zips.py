"""Check Phase 3 submission zips against the test files and copy them for the report."""
import json
import os
import shutil
import sys
import zipfile

import torch

DATA = r"C:\Users\Anshuman Raj\CS781\phase3data\Phase_3_RR"
REPORT = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\capstone_report\phase3"
RUNS = r"C:\Users\Anshuman Raj\CS781\phase3_runs"
SUBMISSIONS = {
    "efficient_lorma": os.path.join(RUNS, "efficient_ens_7_13"),
    "hybrid_lora": os.path.join(RUNS, "hybrid_ens"),
    "decoder_ia3": os.path.join(RUNS, "decoder_ens"),
    "encdec_distill": os.path.join(RUNS, "encdec_distill16"),
    "encdec_distill_s7": os.path.join(RUNS, "encdec_distill_s7"),
    "hybrid_lora_ens": os.path.join(RUNS, "hybrid_ens"),
    "efficient_lorma_3seed": os.path.join(RUNS, "efficient_ens_7_13_21"),
    "efficient_lorma_full": os.path.join(RUNS, "eff_full_7_13"),
    "efficient_lorma_allfull": os.path.join(RUNS, "effA_full_7_13"),
    "efficient_lorma_full_rarity": os.path.join(RUNS, "eff_full_7_13_r05"),
    "hybrid_lora_ens_rarity": os.path.join(RUNS, "hybrid_ens_r05"),
    "decoder_ia3_rarity": os.path.join(RUNS, "decoder_ens_r05"),
    "efficient_lorma_full_ce2_rarity": os.path.join(RUNS, "eff_full_ce2_r05"),
    "hybrid_lora_full": os.path.join(RUNS, "hyb_full_ens"),
    "decoder_ia3_full": os.path.join(RUNS, "dec_full_ens"),
    "encdec_distill_full": os.path.join(RUNS, "encdec_full4"),
}
if len(sys.argv) > 1:
    SUBMISSIONS = {name: SUBMISSIONS[name] for name in sys.argv[1:]}

vocab = set(json.load(open(os.path.join(DATA, "Train", "label_vocab.json"), encoding="utf-8")))
test = {}
for name in ("CL_test.json", "IT_test.json"):
    test.update(json.load(open(os.path.join(DATA, "Test", name), encoding="utf-8")))

for name, folder in SUBMISSIONS.items():
    zip_path = os.path.join(folder, "submission.zip")
    with zipfile.ZipFile(zip_path) as bundle:
        names = bundle.namelist()
        if names != ["prediction.json"]:
            raise SystemExit(f"{name}: zip holds {names}")
        pred = json.loads(bundle.read("prediction.json"))
    if set(pred) != set(test):
        raise SystemExit(f"{name}: ids differ from the test files")
    for doc_id, sentences in test.items():
        if len(pred[doc_id]) != len(sentences):
            raise SystemExit(f"{name}: {doc_id} has {len(pred[doc_id])} roles for {len(sentences)} sentences")
        if any(label not in vocab for label in pred[doc_id]):
            raise SystemExit(f"{name}: {doc_id} has a label outside label_vocab.json")
    best = os.path.join(folder, "best.pt")
    dev_f1 = torch.load(best, map_location="cpu", weights_only=False)["dev_f1"] if os.path.isfile(best) else float("nan")
    dest = os.path.join(REPORT, name)
    os.makedirs(dest, exist_ok=True)
    for file in ("submission.zip", "prediction.json", "dev_prediction.json", "dev_truth_cl.json", "dev_truth_it.json"):
        shutil.copyfile(os.path.join(folder, file), os.path.join(dest, file))
    roles = sum(len(value) for value in pred.values())
    print(f"{name:16s} ok  docs {len(pred)}  roles {roles}  dev macro-F1 {dev_f1:.4f}  -> {dest}\\submission.zip")
