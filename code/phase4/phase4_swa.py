"""Average the two efficient-partial checkpoints and score dev, then write a zip."""
import json
import os
import zipfile

import torch

from legal_bpe import LegalBPE
from phase4_own import (
    DEFAULT_BPE,
    DEFAULT_DATA,
    DEFAULT_WEIGHTS,
    best_cut,
    build_case,
    dump_scores,
    macro_f1,
    positive_scores,
    prepare,
    read_rows,
)

LONG = r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_long\best.pt"
MORE = r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_more\best.pt"
OUT = r"C:\Users\Anshuman Raj\CS781\phase4_runs\efficient_partial_swa"


def averaged(first, second, keep):
    mixed = {}
    for name, value in first.items():
        if name not in second or not torch.is_tensor(value):
            continue
        other = second[name]
        if other.shape != value.shape:
            continue
        mixed[name] = (1.0 - keep) * value.float() + keep * other.float()
    return mixed


def score(state, dev_rows, test_rows, device):
    model = build_case("efficient", "partial", DEFAULT_WEIGHTS["efficient"], 8, 16.0, 10, 0)
    model.load_state_dict(state, strict=False)
    model.to(device)
    dev_scores = positive_scores(model, dev_rows, device)
    gold = [row["label"] for row in dev_rows]
    argmax = macro_f1(gold, [1 if item >= 0.5 else 0 for item in dev_scores])
    test_scores = positive_scores(model, test_rows, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return argmax, dev_scores, test_scores


def write_zip(dev_rows, test_rows, dev_scores, test_scores, argmax):
    os.makedirs(OUT, exist_ok=True)
    dump_scores(os.path.join(OUT, "dev_scores.json"), dev_rows, dev_scores)
    dump_scores(os.path.join(OUT, "test_scores.json"), test_rows, test_scores)
    predictions = [{"id": row["id"], "pred": int(score >= 0.5)} for row, score in zip(test_rows, test_scores)]
    pred_path = os.path.join(OUT, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    zip_path = os.path.join(OUT, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"argmax": argmax, "threshold": 0.5, "chosen": argmax}, handle)
    print("zip", zip_path, len(predictions), flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(DEFAULT_DATA, "Phase_4_CJPE")
    dev_rows = read_rows(os.path.join(root, "Train", "dev.json"), True)
    test_rows = read_rows(os.path.join(root, "Test", "test.json"), False)
    tokenizer = LegalBPE.load(DEFAULT_BPE)
    dev_rows = prepare(dev_rows, tokenizer, 3, 256, 1000, "tail", "tail")
    test_rows = prepare(test_rows, tokenizer, 3, 256, 1000, "tail", "tail")
    first = torch.load(LONG, map_location="cpu", weights_only=False)["model"]
    second = torch.load(MORE, map_location="cpu", weights_only=False)["model"]
    chosen = {}
    for keep in (0.0, 0.35, 0.5):
        state = first if keep == 0.0 else averaged(first, second, keep)
        argmax, dev_scores, test_scores = score(state, dev_rows, test_rows, device)
        print(f"keep_more {keep:.2f} dev argmax {argmax:.4f}", flush=True)
        chosen[keep] = (argmax, dev_scores, test_scores)
    # Stay close to the checkpoint that scored 0.6527. Do not pick the dev maximum.
    argmax, dev_scores, test_scores = chosen[0.35]
    print("using keep_more 0.35", round(argmax, 4), flush=True)
    write_zip(dev_rows, test_rows, dev_scores, test_scores, argmax)


if __name__ == "__main__":
    main()
