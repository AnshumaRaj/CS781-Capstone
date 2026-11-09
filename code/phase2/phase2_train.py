"""Phase 2 legal NER: fine-tune cased BERT and write submission.zip.

A span counts only when the start, the end, and the label all match.
The score is the macro-F1 of the 12 labels. Tagging is done on words whose
edges match those spans, with BERT contextualising each word.
"""

import argparse
import json
import os
import random
import re
import time
import zipfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

BASE = r"C:\Users\Anshuman Raj\.cache\huggingface\hub\datasets--Exploration-Lab--CS-781-Capstone\snapshots\89edba6a44d6bb9593fd13b6895b5cd3b06eb82e\Phase_2_LNER"
MODEL = r"C:\Users\Anshuman Raj\.cache\huggingface\hub\models--google-bert--bert-base-cased\snapshots\cd5ef92a9fb2f889e972770a36d4ed042daf221e"
INLEGAL = r"C:\Users\Anshuman Raj\.cache\huggingface\hub\models--law-ai--InLegalBERT\snapshots\b5ecfed8ed6cf9d25a3cb8225a8c52f161f7401a"
INCASE = r"C:\Users\Anshuman Raj\.cache\huggingface\hub\models--law-ai--InCaseLawBERT\snapshots\7f2c2a0c1ff4149e8c4a8c79ee9f24757ad5dacd"
WORK = r"C:\Users\Anshuman Raj\OneDrive\Desktop\CS781\_lner_work"
LABELS = [
    "APP",
    "RESP",
    "A.COUNSEL",
    "R.COUNSEL",
    "JUDGE",
    "WIT",
    "AUTH",
    "COURT",
    "STAT",
    "PREC",
    "DATE",
    "CASENO",
]
TAG_NAMES = ["O"] + [prefix + lab for lab in LABELS for prefix in ("B-", "I-")]
TAG_INDEX = {name: index for index, name in enumerate(TAG_NAMES)}
WORD = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+|[A-Za-z]+|[0-9]+|[^\s]")
MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}\.?,?\s+\d{{4}}"
    rf"|{MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})\b",
    re.IGNORECASE,
)
MAX_LEN = 256
STRIDE = 128
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(text_path, gold_path=None):
    texts = read_json(text_path)
    golds = read_json(gold_path) if gold_path else {}
    rows = []
    for doc_id, text in texts.items():
        spans = []
        for item in golds.get(doc_id, []):
            start, end, label = int(item[0]), int(item[1]), str(item[2])
            if label in LABELS and 0 <= start < end <= len(text):
                spans.append((start, end, label))
        rows.append((str(doc_id), text, spans))
    return rows


def word_spans(text):
    return [(match.start(), match.end()) for match in WORD.finditer(text)]


def tags_for_words(words, entities):
    assigned = [-1] * len(words)
    for entity_index, (start, end, _label) in enumerate(entities):
        for index, (left, right) in enumerate(words):
            if left >= start and right <= end:
                assigned[index] = entity_index
    tags = []
    for index, entity_index in enumerate(assigned):
        if entity_index < 0:
            tags.append("O")
            continue
        label = entities[entity_index][2]
        previous = assigned[index - 1] if index else -1
        prefix = "I-" if previous == entity_index else "B-"
        tags.append(prefix + label)
    return tags


def transition_matrix():
    """Legal BIO transitions. Illegal I- tags are blocked; a legal continuation is preferred."""
    size = len(TAG_NAMES)
    score = torch.zeros(size, size)

    def kind(index):
        if index == 0:
            return "O", ""
        name = TAG_NAMES[index]
        return name[0], name[2:]

    for prev in range(size):
        prev_kind, prev_label = kind(prev)
        for curr in range(size):
            curr_kind, curr_label = kind(curr)
            if curr_kind == "I" and not (prev_kind in ("B", "I") and prev_label == curr_label):
                score[prev, curr] = -12.0
            elif curr_kind == "I":
                score[prev, curr] = 0.6
            elif curr_kind == "B" and prev_kind in ("B", "I") and prev_label == curr_label:
                score[prev, curr] = -0.6
    return score


def viterbi(emissions, transitions):
    """emissions: [words, tags]."""
    length, tags = emissions.shape
    if length == 0:
        return []
    transitions = transitions.to(emissions.device)
    back = torch.zeros(length, tags, dtype=torch.long)
    chart = emissions[0].clone()
    for index in range(1, length):
        scores = chart.unsqueeze(1) + transitions + emissions[index].unsqueeze(0)
        best, back[index] = scores.max(dim=0)
        chart = best
    path = [0] * length
    path[-1] = int(chart.argmax())
    for index in range(length - 2, -1, -1):
        path[index] = int(back[index + 1, path[index + 1]])
    return [TAG_NAMES[index] for index in path]


def spans_from_word_tags(words, tags):
    found = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if not tag.startswith("B-") and not tag.startswith("I-"):
            index += 1
            continue
        label = tag[2:]
        cursor = index + 1
        while cursor < len(tags) and tags[cursor] == "I-" + label:
            cursor += 1
        start = words[index][0]
        end = words[cursor - 1][1]
        if end > start:
            found.append((start, end, label))
        index = cursor
    unique = []
    seen = set()
    for span in found:
        if span not in seen:
            seen.add(span)
            unique.append(span)
    return unique


def macro_f1(rows, predictions):
    stats = {label: [0, 0, 0] for label in LABELS}
    for doc_id, _text, entities in rows:
        gold = {(start, end, label) for start, end, label in entities}
        pred = set(predictions.get(doc_id, []))
        for label in LABELS:
            gold_label = {span for span in gold if span[2] == label}
            pred_label = {span for span in pred if span[2] == label}
            stats[label][0] += len(gold_label & pred_label)
            stats[label][1] += len(pred_label - gold_label)
            stats[label][2] += len(gold_label - pred_label)
    scores = []
    lines = []
    for label in LABELS:
        true_pos, false_pos, false_neg = stats[label]
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
        lines.append(
            f"{label:12} P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
            f"tp={true_pos} fp={false_pos} fn={false_neg}"
        )
    return sum(scores) / len(scores), lines


def encode_document(tokenizer, text, entities):
    words = word_spans(text)
    tags = tags_for_words(words, entities)
    encoded = tokenizer(
        text,
        max_length=MAX_LEN,
        stride=STRIDE,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], int):
        input_ids = [input_ids]
        masks = [encoded["attention_mask"]]
        offsets = [encoded["offset_mapping"]]
    else:
        masks = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]
    starts = {left: index for index, (left, _right) in enumerate(words)}
    windows = []
    for ids, mask, offset in zip(input_ids, masks, offsets):
        labels = [-100] * len(ids)
        for token_index, (start, end) in enumerate(offset):
            if not mask[token_index] or start == end:
                continue
            word_index = starts.get(start)
            if word_index is None:
                continue
            labels[token_index] = TAG_INDEX[tags[word_index]]
        windows.append({"ids": ids, "mask": mask, "labels": labels, "offset": offset})
    return words, tags, windows


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        return (
            torch.tensor(window["ids"], dtype=torch.long),
            torch.tensor(window["mask"], dtype=torch.long),
            torch.tensor(window["labels"], dtype=torch.long),
        )


def class_weights(windows):
    counts = torch.ones(len(TAG_NAMES))
    for window in windows:
        for label in window["labels"]:
            if label >= 0:
                counts[label] += 1
    weights = counts.sum() / (counts * len(TAG_NAMES))
    weights = torch.clamp(weights, max=8.0)
    return weights / weights.mean()


def oversample(windows):
    rare = {
        TAG_INDEX["B-" + label]
        for label in ("APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT")
    }
    chosen = []
    for window in windows:
        chosen.append(window)
        if any(label in rare for label in window["labels"]):
            chosen.append(window)
    return chosen


def oracle_score(tokenizer, rows):
    predictions = {}
    for doc_id, text, entities in rows:
        words, tags, _windows = encode_document(tokenizer, text, entities)
        predictions[doc_id] = spans_from_word_tags(words, tags)
    return macro_f1(rows, predictions)


def build_model(model_path):
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(TAG_NAMES),
        id2label={index: name for index, name in enumerate(TAG_NAMES)},
        label2id=TAG_INDEX,
    )
    model.gradient_checkpointing_enable()
    return model


def run_epoch(model, loader, optimizer, weights, device, train):
    model.train(train)
    total = 0.0
    seen = 0
    for input_ids, mask, labels in loader:
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            loss = F.cross_entropy(
                logits.float().view(-1, len(TAG_NAMES)),
                labels.view(-1),
                weight=weights,
                ignore_index=-100,
            )
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * input_ids.size(0)
        seen += input_ids.size(0)
    return total / max(seen, 1)


@torch.no_grad()
def cache_emissions(model, tokenizer, rows, device, batch_size):
    """Average word logits for every document. Used to tune the decoder."""
    model.eval()
    cached = {}
    for doc_id, text, entities in rows:
        words, _tags, windows = encode_document(tokenizer, text, entities)
        summed = torch.zeros(len(words), len(TAG_NAMES))
        counts = torch.zeros(len(words))
        starts = {left: index for index, (left, _right) in enumerate(words)}
        loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False)
        cursor = 0
        for input_ids, mask, _labels in loader:
            width = input_ids.size(0)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits.float().cpu()
            batch_mask = mask
            for row in range(width):
                window = windows[cursor + row]
                for token_index, (start, end) in enumerate(window["offset"]):
                    if not batch_mask[row, token_index] or start == end:
                        continue
                    word_index = starts.get(start)
                    if word_index is None:
                        continue
                    summed[word_index] += logits[row, token_index]
                    counts[word_index] += 1
            cursor += width
        averaged = summed / counts.clamp(min=1).unsqueeze(1)
        cached[doc_id] = (words, averaged, counts)
    return cached


def spans_from_emissions(words, averaged, counts, minimum):
    """Viterbi, then drop a span whose least-confident word is below `minimum`."""
    present = counts > 0
    emissions = averaged.clone()
    emissions[~present] = 0
    tags = viterbi(emissions, transition_matrix())
    for index in range(len(words)):
        if not present[index]:
            tags[index] = "O"
    probs = torch.softmax(averaged, dim=-1)
    found = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if not tag.startswith("B-") and not tag.startswith("I-"):
            index += 1
            continue
        label = tag[2:]
        cursor = index + 1
        while cursor < len(tags) and tags[cursor] == "I-" + label:
            cursor += 1
        scores = [float(probs[word, TAG_INDEX[tags[word]]]) for word in range(index, cursor)]
        start = words[index][0]
        end = words[cursor - 1][1]
        if end > start and min(scores) >= minimum.get(label, 0.0):
            found.append((start, end, label))
        index = cursor
    unique = []
    seen = set()
    for span in found:
        if span not in seen:
            seen.add(span)
            unique.append(span)
    return unique


def predictions_from_cache(rows, cached, minimum):
    return {
        doc_id: spans_from_emissions(*cached[doc_id], minimum)
        for doc_id, _text, _entities in rows
    }


@torch.no_grad()
def predict_rows(model, tokenizer, rows, device, batch_size):
    model.eval()
    predictions = {}
    for doc_id, text, entities in rows:
        words, _tags, windows = encode_document(tokenizer, text, entities)
        summed = torch.zeros(len(words), len(TAG_NAMES))
        counts = torch.zeros(len(words))
        starts = {left: index for index, (left, _right) in enumerate(words)}
        loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False)
        cursor = 0
        for input_ids, mask, _labels in loader:
            width = input_ids.size(0)
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=mask).logits.float().cpu()
            batch_mask = mask.cpu()
            for row in range(width):
                window = windows[cursor + row]
                for token_index, (start, end) in enumerate(window["offset"]):
                    if not batch_mask[row, token_index] or start == end:
                        continue
                    word_index = starts.get(start)
                    if word_index is None:
                        continue
                    summed[word_index] += logits[row, token_index]
                    counts[word_index] += 1
            cursor += width
        averaged = summed / counts.clamp(min=1).unsqueeze(1)
        tags = viterbi(averaged, transition_matrix())
        for index in range(len(words)):
            if counts[index] == 0:
                tags[index] = "O"
        predictions[doc_id] = spans_from_word_tags(words, tags)
    return predictions


def date_spans(text):
    return [(match.start(), match.end(), "DATE") for match in DATE_RE.finditer(text)]


def overlaps(start, end, spans):
    return any(not (end <= left or start >= right) for left, right, _label in spans)


NAME_RE = r"[A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+){0,5}"
WIT_RES = [
    re.compile(rf"({NAME_RE})\s*\(\s*P\.?\s*W\.?\s*[-–]?\s*\d+", re.I),
    re.compile(rf"({NAME_RE})\s+(?:was|were)\s+examined\s+as\s+P\.?\s*W", re.I),
    re.compile(rf"P\.?\s*W\.?\s*[-–]?\s*\d+\s*[-–,]\s*({NAME_RE})", re.I),
]
CITE_RE = re.compile(r"(?i)(?:SCC|AIR|SCR|JT|SCALE|Supp|Cri\.?\s*L\.?\s*J)")
STOP_SURF = {
    "the", "of", "and", "a", "an", "in", "to", "for", "by", "or", "on", "at",
    "from", "with", "this", "that", "be", "is", "was", "were", "as", "it",
    "its", "no", "vs", "v", "bench", "bench:",
}


def trim_span(text, start, end, label):
    """Cut footnote numbers, '(supra)', and a trailing citation the gold leaves off."""
    if label not in ("PREC", "CASENO", "COURT", "STAT", "AUTH"):
        return start, end
    for _ in range(6):
        if end <= start:
            break
        surface = text[start:end]
        if surface.endswith("(supra)") or surface.endswith("(supra"):
            end = start + surface.rfind("(supra")
        elif re.search(r"\s+reported in\b", surface, re.I):
            end = start + re.search(r"\s+reported in\b", surface, re.I).start()
        elif label == "PREC" and re.search(r"\.\d{1,2}$", surface) and not CITE_RE.search(surface[-30:]):
            end -= len(re.search(r"\d{1,2}$", surface).group())
        elif label == "PREC" and re.search(r"\.\s+\d{1,2}$", surface) and not CITE_RE.search(surface[-40:]):
            end -= len(re.search(r"\s+\d{1,2}$", surface).group())
        elif surface.endswith(")") and surface.count(")") > surface.count("("):
            end -= 1
        else:
            break
        while end > start and text[end - 1] in " ,;":
            end -= 1
    return start, end


def witness_names(text, spans):
    found = set()
    for pattern in WIT_RES:
        for match in pattern.finditer(text):
            name = " ".join(match.group(1).split()).casefold()
            if len(name) >= 6:
                found.add(name)
    for start, end, label in spans:
        if label != "WIT":
            continue
        name = " ".join(text[start:end].split()).casefold()
        if len(name) >= 6:
            found.add(name)
    return found


def polish(rows, predictions):
    """Trim precedent edges and copy a witness name onto its other mentions."""
    person = {"APP", "RESP", "AUTH", "PREC", "JUDGE", "A.COUNSEL", "R.COUNSEL", "WIT"}
    polished = {}
    for doc_id, text, _entities in rows:
        spans = []
        seen = set()
        for start, end, label in predictions.get(doc_id, []):
            if label == "JUDGE" and text[start:end].endswith(" BENCH:"):
                end -= len(" BENCH:")
            start, end = trim_span(text, start, end, label)
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end <= start or text[start:end].casefold() in STOP_SURF:
                continue
            span = (start, end, label)
            if span not in seen:
                seen.add(span)
                spans.append(span)
        seeds = witness_names(text, spans)
        relabeled = []
        for start, end, label in spans:
            name = " ".join(text[start:end].split()).casefold()
            if name in seeds and label in person:
                label = "WIT"
            relabeled.append((start, end, label))
        polished[doc_id] = relabeled
    return polished


def add_dates(rows, predictions):
    """Drop model dates and use the high-precision date patterns instead."""
    merged = {}
    for doc_id, text, _entities in rows:
        dates = date_spans(text)
        spans = []
        for start, end, label in predictions.get(doc_id, []):
            if label == "DATE":
                continue
            if overlaps(start, end, dates):
                continue
            spans.append((start, end, label))
        spans.extend(dates)
        merged[doc_id] = spans
    return merged


def train_model(tokenizer, train_rows, dev_rows, device, epochs, batch_size, out_path, model_path):
    windows = []
    for doc_id, text, entities in train_rows:
        _words, _tags, doc_windows = encode_document(tokenizer, text, entities)
        windows.extend(doc_windows)
    windows = oversample(windows)
    weights = class_weights(windows).to(device)
    print(f"train windows {len(windows)}", flush=True)
    loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=True)
    model = build_model(model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    best_f1 = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        loss = run_epoch(model, loader, optimizer, weights, device, train=True)
        message = f"epoch {epoch} loss {loss:.4f} time {time.time() - started:.0f}s"
        if dev_rows:
            predictions = predict_rows(model, tokenizer, dev_rows, device, batch_size)
            score, lines = macro_f1(dev_rows, predictions)
            dated = add_dates(dev_rows, predictions)
            dated_score, dated_lines = macro_f1(dev_rows, dated)
            message += f"  dev {score:.4f}  dev+date {dated_score:.4f}"
            chosen = dated_score if dated_score >= score else score
            chosen_lines = dated_lines if dated_score >= score else lines
            if chosen > best_f1:
                best_f1 = chosen
                best_epoch = epoch
                torch.save(model.state_dict(), out_path)
                print(message + "  *", flush=True)
                for line in chosen_lines:
                    print("   ", line, flush=True)
            else:
                print(message, flush=True)
        else:
            print(message, flush=True)
            torch.save(model.state_dict(), out_path)
            best_epoch = epoch
            best_f1 = 0.0
    if dev_rows and os.path.isfile(out_path):
        model.load_state_dict(torch.load(out_path, map_location=device))
    return model, best_epoch, best_f1


def write_zip(folder, predictions):
    os.makedirs(folder, exist_ok=True)
    payload = {
        doc_id: [[start, end, label] for start, end, label in spans]
        for doc_id, spans in predictions.items()
    }
    pred_path = os.path.join(folder, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    zip_path = os.path.join(folder, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["oracle", "cv", "final", "predict"], default="cv")
    parser.add_argument("--model", choices=["bert", "inlegal", "incase"], default="inlegal")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--folder", default="")
    parser.add_argument("--weights", default="")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    set_seed(args.seed)
    print("seed", args.seed, flush=True)
    os.makedirs(WORK, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    model_path = {"bert": MODEL, "inlegal": INLEGAL, "incase": INCASE}[args.model]
    print("model", args.model, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dir = os.path.join(BASE, "Train")
    fold1 = load_rows(
        os.path.join(train_dir, "fold1-text.json"),
        os.path.join(train_dir, "fold1-gold.json"),
    )
    fold2 = load_rows(
        os.path.join(train_dir, "fold2-text.json"),
        os.path.join(train_dir, "fold2-gold.json"),
    )
    test_rows = load_rows(os.path.join(BASE, "Test", "fold3-text.json"))
    print("docs", len(fold1), len(fold2), len(test_rows), flush=True)
    if args.mode == "oracle":
        score, lines = oracle_score(tokenizer, fold1 + fold2)
        print(f"oracle macro-F1 {score:.4f}", flush=True)
        for line in lines:
            print(line, flush=True)
        return
    if args.mode == "cv":
        train_model(
            tokenizer,
            fold1,
            fold2,
            device,
            args.epochs,
            args.batch,
            os.path.join(WORK, f"cv-{args.model}-s{args.seed}.pt"),
            model_path,
        )
        return
    if args.mode == "predict":
        model = build_model(model_path).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        model, _epoch, _score = train_model(
            tokenizer,
            fold1 + fold2,
            [],
            device,
            args.epochs,
            args.batch,
            os.path.join(WORK, f"final-{args.model}-s{args.seed}.pt"),
            model_path,
        )
    predictions = predict_rows(model, tokenizer, test_rows, device, args.batch)
    predictions = add_dates(test_rows, predictions)
    for doc_id, text, _entities in test_rows:
        clean = []
        seen = set()
        for start, end, label in sorted(predictions[doc_id]):
            if label == "JUDGE" and text[start:end].endswith(" BENCH:"):
                end -= len(" BENCH:")
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end <= start or end > len(text) or text[start:end].casefold() in STOP_SURF or label not in LABELS:
                continue
            span = (start, end, label)
            if span not in seen:
                seen.add(span)
                clean.append(span)
        predictions[doc_id] = clean
    folder = args.folder
    zip_path = write_zip(folder, predictions)
    counts = {label: 0 for label in LABELS}
    for spans in predictions.values():
        for _start, _end, label in spans:
            counts[label] += 1
    print("zip", zip_path, flush=True)
    print("spans", counts, flush=True)


if __name__ == "__main__":
    main()
