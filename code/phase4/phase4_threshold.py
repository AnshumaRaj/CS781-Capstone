import json
import os
import zipfile

import torch

from phase4_own import DEFAULT_BPE, DEFAULT_DATA, LegalBPE, build_case, macro_f1, prepare, read_rows

RUNS = r"C:\Users\Anshuman Raj\CS781\phase4_runs"
DEST = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\p4_sub"
JOBS = {
    "efficient_lorma": (4, 96),
    "hybrid_adapter": (6, 128),
    "decoder_bitfit": (6, 128),
    "encdec_partial": (6, 128),
    "efficient_partial": (6, 128),
    "encdec_distill": (6, 128),
}


def scores(model, rows, device):
    values = []
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            batch = rows[start : start + 16]
            logits = model.logits_many([row["windows"] for row in batch], device).float()
            values.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return values


def best_threshold(gold, positive):
    best_f1, best_cut = -1.0, 0.5
    cuts = sorted(set(positive))
    candidates = [0.5]
    for left, right in zip(cuts, cuts[1:]):
        candidates.append((left + right) / 2)
    candidates.extend([min(cuts) - 1e-4, max(cuts) + 1e-4])
    for cut in candidates:
        pred = [1 if score >= cut else 0 for score in positive]
        f1 = macro_f1(gold, pred)
        if f1 > best_f1:
            best_f1, best_cut = f1, cut
    return best_cut, best_f1


def sweep(name, chunks, width):
    folder = os.path.join(RUNS, name)
    checkpoint_path = os.path.join(folder, "best.pt")
    if not os.path.exists(checkpoint_path):
        print("skip", name, flush=True)
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_case(
        checkpoint["arch"], checkpoint["method"], checkpoint["weights"], checkpoint["rank"], checkpoint["alpha"]
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device).eval()
    tokenizer = LegalBPE.load(DEFAULT_BPE)
    root = os.path.join(DEFAULT_DATA, "Phase_4_CJPE")
    dev_rows = prepare(read_rows(os.path.join(root, "Train", "dev.json"), True), tokenizer, chunks, width, 6000)
    test_rows = prepare(read_rows(os.path.join(root, "Test", "test.json"), False), tokenizer, chunks, width, 6000)
    dev_score = scores(model, dev_rows, device)
    test_score = scores(model, test_rows, device)
    cut, f1 = best_threshold([row["label"] for row in dev_rows], dev_score)
    predictions = [
        {"id": row["id"], "pred": int(score >= cut)}
        for row, score in zip(test_rows, test_score)
    ]
    used = sorted({row["pred"] for row in predictions})
    print(name, "threshold", round(cut, 4), "dev macro-F1", round(f1, 4), "test preds", used, flush=True)
    pred_path = os.path.join(folder, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False)
    zip_path = os.path.join(folder, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    out = os.path.join(DEST, name)
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for job, shape in JOBS.items():
        sweep(job, *shape)
