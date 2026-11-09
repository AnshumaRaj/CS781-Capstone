"""Test whether weaker efficient seeds raise the seed-42 + seed-7 average.

Logits are cached one model at a time on the GPU. A test zip is written only
when some average beats the held-out score of seed 42 + seed 7.
"""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781")

import torch

import phase2_lner as p

DATA = r"C:\Users\Anshuman Raj\CS781\kaggle_250m_out\phase2data"
PHASE1 = p.DEFAULT_WEIGHTS["efficient"]
BPE = p.DEFAULT_BPE
OUT = r"C:\Users\Anshuman Raj\CS781\phase2_runs\ensemble_weak"
WIDTH = 512
STRIDE = 256
MODELS = {
    "s42": r"C:\Users\Anshuman Raj\CS781\kaggle_full_crf_v2\efficient_full_crf\best.pt",
    "s7": r"C:\Users\Anshuman Raj\CS781\kaggle_seed7_out\efficient_full_s7\best.pt",
    "s13": r"C:\Users\Anshuman Raj\CS781\phase2_runs\efficient_full_s13\best.pt",
    "s3": r"C:\Users\Anshuman Raj\CS781\kaggle_seed3_out\efficient_full_s3\best.pt",
    "s21": r"C:\Users\Anshuman Raj\CS781\kaggle_seed21_out\efficient_full_s21\best.pt",
}
TRIALS = [
    ("s42+s7", {"s42": 1.0, "s7": 1.0}),
    ("plus-s13", {"s42": 1.0, "s7": 1.0, "s13": 1.0}),
    ("plus-s3", {"s42": 1.0, "s7": 1.0, "s3": 1.0}),
    ("plus-s21", {"s42": 1.0, "s7": 1.0, "s21": 1.0}),
    ("s13-half", {"s42": 1.0, "s7": 1.0, "s13": 0.5}),
    ("s3-half", {"s42": 1.0, "s7": 1.0, "s3": 0.5}),
    ("s21-half", {"s42": 1.0, "s7": 1.0, "s21": 0.5}),
    ("plus-s13-s3", {"s42": 1.0, "s7": 1.0, "s13": 1.0, "s3": 1.0}),
    ("all-equal", {"s42": 1.0, "s7": 1.0, "s13": 1.0, "s3": 1.0, "s21": 1.0}),
    ("weak-half", {"s42": 1.0, "s7": 1.0, "s13": 0.5, "s3": 0.5, "s21": 0.5}),
    ("weak-quarter", {"s42": 1.0, "s7": 1.0, "s13": 0.25, "s3": 0.25, "s21": 0.25}),
]


def load_tagger(path, device):
    names = p.tag_names_of()
    tagger = p.build_tagger("efficient", "full", 8192, len(names), PHASE1, 8, 16.0, names, True)
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    tagger.load_state_dict(state)
    p.enable_full_attention(tagger.backbone)
    tagger.to(device)
    tagger.eval()
    return tagger


def token_logits(tagger, tokenizer, text, device):
    names = p.tag_names_of()
    ids, spans = tokenizer.encode_with_offsets(text)
    if not ids:
        return None, None
    summed = torch.zeros(len(ids), len(names))
    counts = torch.zeros(len(ids), 1)
    for window in p.make_windows(ids, spans, ["O"] * len(ids), WIDTH, STRIDE):
        token_ids = torch.tensor([window["ids"]], dtype=torch.long, device=device)
        pad_mask = torch.ones(1, len(window["ids"]), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = tagger(token_ids, pad_mask)[0].float().cpu()
        begin = window["offset"]
        summed[begin : begin + len(window["ids"])] += logits
        counts[begin : begin + len(window["ids"])] += 1
    return summed / counts.clamp(min=1), spans


def cache_model(key, rows, tokenizer, device, split):
    folder = os.path.join(OUT, split, key)
    os.makedirs(folder, exist_ok=True)
    pending = [row for row in rows if not os.path.isfile(os.path.join(folder, row[0] + ".pt"))]
    if not pending:
        print("cached", split, key, flush=True)
        return
    print("load", key, "docs", len(pending), flush=True)
    started = time.time()
    tagger = load_tagger(MODELS[key], device)
    for index, (doc_id, text, _entities) in enumerate(pending):
        logits, spans = token_logits(tagger, tokenizer, text, device)
        torch.save({"logits": logits, "spans": spans}, os.path.join(folder, doc_id + ".pt"))
        print(f"  {split} {key} {index + 1}/{len(pending)} {doc_id} {time.time() - started:.0f}s", flush=True)
    del tagger
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_blob(split, key, doc_id):
    return torch.load(os.path.join(OUT, split, key, doc_id + ".pt"), map_location="cpu", weights_only=False)


def spans_of(decoder, text, averaged):
    names = p.tag_names_of()
    emissions = averaged.unsqueeze(0)
    mask = torch.ones(1, averaged.size(0), dtype=torch.bool)
    chosen = decoder.crf.decode(emissions, mask)[0].tolist()
    probabilities = (
        averaged.softmax(dim=-1).gather(1, torch.tensor(chosen).unsqueeze(1)).squeeze(1).tolist()
    )
    tags = [names[index] for index in chosen]
    found = p.spans_from_tags(
        text, None, tags, probabilities, {label: 0.0 for label in p.LABELS}
    )
    return found


def predict(decoder, rows, weights, split):
    predictions = {}
    keys = list(weights)
    total_weight = sum(weights.values())
    for doc_id, text, _entities in rows:
        blobs = [load_blob(split, key, doc_id) for key in keys]
        if blobs[0]["logits"] is None:
            predictions[doc_id] = []
            continue
        averaged = sum(blob["logits"] * weights[key] for key, blob in zip(keys, blobs)) / total_weight
        # spans come from the first cached model; tokenization does not depend on the seed
        token_spans = blobs[0]["spans"]
        names = p.tag_names_of()
        emissions = averaged.unsqueeze(0)
        mask = torch.ones(1, averaged.size(0), dtype=torch.bool)
        chosen = decoder.crf.decode(emissions, mask)[0].tolist()
        probabilities = (
            averaged.softmax(dim=-1).gather(1, torch.tensor(chosen).unsqueeze(1)).squeeze(1).tolist()
        )
        tags = [names[index] for index in chosen]
        found = p.spans_from_tags(text, token_spans, tags, probabilities, {label: 0.0 for label in p.LABELS})
        if p.PROPAGATE:
            found = p.propagate_spans(text, found)
        predictions[doc_id] = p.clean_edges(text, found)
    return predictions


def score_trials(decoder, rows, split):
    results = []
    for propagate in (True, False):
        p.PROPAGATE = propagate
        for name, weights in TRIALS:
            if any(not os.path.isfile(os.path.join(OUT, split, key, rows[0][0] + ".pt")) for key in weights):
                continue
            predictions = predict(decoder, rows, weights, split)
            f1, lines = p.macro_f1(rows, predictions)
            print(f"{name} propagate={propagate} {f1:.4f}", flush=True)
            if propagate:
                print("\n".join(lines), flush=True)
            results.append((f1, propagate, name, weights, predictions if split == "dev" else None))
    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    p.DATA_DIR = DATA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    fold1 = p.load_fold("Phase_2_LNER/Train/fold1-text.json", "Phase_2_LNER/Train/fold1-gold.json")
    fold2 = p.load_fold("Phase_2_LNER/Train/fold2-text.json", "Phase_2_LNER/Train/fold2-gold.json")
    _train, dev_rows = p.split_documents(fold1, fold2, 8)
    tokenizer = p.LegalBPE.load(BPE)
    for key in ("s42", "s7"):
        cache_model(key, dev_rows, tokenizer, device, "dev")
    print("load decoder", flush=True)
    decoder = load_tagger(MODELS["s7"], torch.device("cpu"))
    base = score_trials(decoder, dev_rows, "dev")
    base_f1 = max(item[0] for item in base if item[2] == "s42+s7")
    print("s42+s7 best", f"{base_f1:.4f}", flush=True)
    if base_f1 < 0.60:
        print("reproduction is too far from 0.6117, stop", flush=True)
        return
    for key in ("s13", "s3", "s21"):
        cache_model(key, dev_rows, tokenizer, device, "dev")
    results = score_trials(decoder, dev_rows, "dev")
    best_f1, best_prop, best_name, best_weights, _predictions = max(results, key=lambda item: item[0])
    print("best", best_name, "propagate", best_prop, f"{best_f1:.4f}", flush=True)
    if best_f1 <= base_f1 + 0.002:
        print("weaker seeds do not beat seed 42 + seed 7", flush=True)
        return
    p.PROPAGATE = best_prop
    test_rows = p.load_fold("Phase_2_LNER/Test/fold3-text.json")
    for key in best_weights:
        cache_model(key, test_rows, tokenizer, device, "test")
    test_pred = predict(decoder, test_rows, best_weights, "test")
    p.write_submission(os.path.join(OUT, "submission.zip"), test_pred)
    print("wrote", best_name, OUT, flush=True)


if __name__ == "__main__":
    main()
