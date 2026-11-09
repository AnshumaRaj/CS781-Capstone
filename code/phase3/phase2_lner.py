"""Phase 2 legal named-entity recognition, starting from the Phase 1 models.

The score is strict macro-F1. A span counts only when the start character,
the end character, and the label all match. The previous draft of this file
trained a word BiLSTM and stopped at 400 words, so entities later in a
judgment were never predicted. This version:

  * loads one of the four Phase 1 models and adds a token-classification head
  * aligns the Phase 1 byte-level BPE to character offsets
  * slides a window across the whole document, including every test id
  * trains with full fine-tuning, LoRA, LoRMA+, or knowledge distillation

LoRMA+ is the multiplicative adapter from Bihany, Patel, and Modi (2025):

    h = (I + (alpha / r) B A) W x

with B initialised at zero, so the first step matches the Phase 1 model.

The four required submissions, each from a different Phase 1 architecture:

    python phase2_lner.py --arch efficient --method lorma
    python phase2_lner.py --arch decoder --method full
    python phase2_lner.py --arch hybrid --method lora
    python phase2_lner.py --arch encdec --method distill --teacher <efficient checkpoint>

Official deadline: 11 October 2026, 11:59 PM.
"""

import argparse
import json
import os
import random
import shutil
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from legal_bpe import LegalBPE
from phase1_models import EncoderDecoderLM, build_model


SEED = 42
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
LOCAL = r"C:\Users\Anshuman Raj\CS781"
DEFAULT_WEIGHTS = {
    "efficient": os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\model.pt"),
    "decoder": os.path.join(LOCAL, r"local_weights\cs781_phase1_decoder\model.pt"),
    "hybrid": os.path.join(LOCAL, r"local_weights\cs781_phase1_hybrid\model.pt"),
    "encdec": os.path.join(LOCAL, r"encdec_run2\cs781_phase1_encdec\model.pt"),
}
DEFAULT_BPE = os.path.join(LOCAL, r"kaggle_efficient2\cs781_phase1_efficient\legal_bpe.json")
REPO = "Exploration-Lab/CS-781-Capstone"
DATA_DIR = None
PROPAGATE = False
_GENERIC = {
    "state",
    "court",
    "india",
    "union",
    "government",
    "appellant",
    "appellants",
    "respondent",
    "respondents",
    "petitioner",
    "petitioners",
    "others",
    "other",
    "ors",
    "anr",
    "the",
    "shri",
    "smt",
    "and",
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fetch(name):
    if DATA_DIR:
        local = os.path.join(DATA_DIR, *name.split("/"))
        if os.path.isfile(local):
            print("local", name, flush=True)
            return local
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset")
    print("got", name, flush=True)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_fold(text_name, gold_name=None):
    texts = read_json(fetch(text_name))
    golds = read_json(fetch(gold_name)) if gold_name else {}
    rows = []
    for doc_id, text in texts.items():
        spans = []
        for item in golds.get(doc_id, []):
            start, end, label = int(item[0]), int(item[1]), str(item[2])
            if label in LABELS and 0 <= start < end <= len(text):
                spans.append((start, end, label))
        rows.append((str(doc_id), text, spans))
    return rows


def tag_names_of():
    names = ["O"]
    for label in LABELS:
        names.append("B-" + label)
        names.append("I-" + label)
    return names


def bio_tags(token_spans, entities):
    """Label each token with the entity that covers the most of its characters."""
    assigned = [-1] * len(token_spans)
    for index, (left, right) in enumerate(token_spans):
        best = -1
        best_overlap = 0
        for entity_index, (start, end, _label) in enumerate(entities):
            overlap = min(right, end) - max(left, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = entity_index
        if best_overlap > 0:
            assigned[index] = best
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


def make_windows(ids, spans, tags, width, stride):
    rows = []
    start = 0
    while start < len(ids):
        stop = min(start + width, len(ids))
        rows.append(
            {
                "ids": ids[start:stop],
                "spans": spans[start:stop],
                "tags": tags[start:stop],
                "offset": start,
            }
        )
        if stop == len(ids):
            break
        start += stride
    return rows


def prepare_document(tokenizer, doc_id, text, entities, width, stride, tag_index):
    ids, spans = tokenizer.encode_with_offsets(text)
    tags = bio_tags(spans, entities)
    windows = make_windows(ids, spans, tags, width, stride)
    for window in windows:
        window["doc_id"] = doc_id
        window["target"] = [tag_index[tag] for tag in window["tags"]]
    return ids, spans, tags, windows


def boundary_report(tokenizer, rows):
    """How often a gold span lands exactly on token boundaries."""
    covered = 0
    total = 0
    tiled = 0
    shown = 0
    for _doc_id, text, entities in rows:
        ids, spans = tokenizer.encode_with_offsets(text)
        if spans and spans[0][0] == 0 and spans[-1][1] == len(text):
            if all(right == next_left for (_left, right), (next_left, _next_right) in zip(spans, spans[1:])):
                tiled += 1
        starts = {left for left, _right in spans}
        ends = {right for _left, right in spans}
        for start, end, label in entities:
            total += 1
            if start in starts and end in ends:
                covered += 1
            elif shown < 5:
                shown += 1
                print(
                    "unaligned",
                    label,
                    repr(text[start:end][:60]),
                    "around",
                    repr(text[max(0, start - 8) : end + 8][:80]),
                )
    print(
        f"documents tiled by tokens {tiled}/{len(rows)}  "
        f"gold spans on token boundaries {covered}/{total}",
        flush=True,
    )
    return covered, total


class LoRALinear(nn.Module):
    """Additive low-rank update. The base Phase 1 matrix stays frozen."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        out_features, in_features = base.weight.shape
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        update = (x @ self.A.transpose(0, 1)) @ self.B.transpose(0, 1)
        return self.base(x) + self.scale * update


class LoRMALinear(nn.Module):
    """LoRMA+ : h = (I + (alpha / r) B A) W x, with B starting at zero."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        out_features, _in_features = base.weight.shape
        self.A = nn.Parameter(torch.randn(rank, out_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        hidden = self.base(x)
        update = (hidden @ self.A.transpose(0, 1)) @ self.B.transpose(0, 1)
        return hidden + self.scale * update


def adapt_linears(model, kind, rank, alpha):
    wrapper = LoRMALinear if kind == "lorma" else LoRALinear
    count = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, wrapper(child, rank, alpha))
                count += 1
    return count


def enable_full_attention(model):
    """Let a causal Phase 1 model read tokens on both sides while tagging.

    NER needs the words after a name. The language-model weights stay as
    trained; only the attention mask used for this task is removed.
    """
    count = 0
    for module in model.modules():
        if getattr(module, "causal", None) is not None and hasattr(module, "bidirectional"):
            module.bidirectional = True
            count += 1
    return count


class ChainCRF(nn.Module):
    """Linear-chain CRF over BIO tags. Illegal transitions stay blocked."""

    def __init__(self, tag_names):
        super().__init__()
        n_tags = len(tag_names)
        self.transitions = nn.Parameter(torch.zeros(n_tags, n_tags))
        allowed = torch.zeros(n_tags, n_tags, dtype=torch.bool)
        for src, src_name in enumerate(tag_names):
            for dst, dst_name in enumerate(tag_names):
                if dst_name == "O" or dst_name.startswith("B-"):
                    allowed[src, dst] = True
                else:
                    label = dst_name[2:]
                    allowed[src, dst] = src_name in ("B-" + label, "I-" + label)
        self.register_buffer("allowed", allowed)
        self.register_buffer("score_bias", torch.zeros(n_tags))
        with torch.no_grad():
            self.transitions.masked_fill_(~allowed, -10.0)

    def transition(self):
        return self.transitions.masked_fill(~self.allowed, -1e4)

    def _emit(self, emissions):
        return emissions + self.score_bias.view(1, 1, -1)

    def nll(self, emissions, tags, mask):
        emissions = self._emit(emissions)
        safe = tags.clamp(min=0)
        gold = self._path_score(emissions, safe, mask)
        logz = self._partition(emissions, mask)
        lengths = mask.float().sum(dim=1).clamp(min=1)
        return ((logz - gold) / lengths).mean()

    def _path_score(self, emissions, tags, mask):
        trans = self.transition()
        score = emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for step in range(1, emissions.size(1)):
            jump = trans[tags[:, step - 1], tags[:, step]]
            token = emissions[:, step].gather(1, tags[:, step : step + 1]).squeeze(1)
            score = score + (jump + token) * mask[:, step].float()
        return score

    def _partition(self, emissions, mask):
        trans = self.transition()
        alpha = emissions[:, 0]
        for step in range(1, emissions.size(1)):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, step].unsqueeze(1)
            nxt = torch.logsumexp(scores, dim=1)
            active = mask[:, step].unsqueeze(1)
            alpha = torch.where(active, nxt, alpha)
        return torch.logsumexp(alpha, dim=-1)

    def decode(self, emissions, mask):
        emissions = self._emit(emissions)
        trans = self.transition()
        batch, steps, n_tags = emissions.shape
        back = torch.zeros(batch, steps, n_tags, dtype=torch.long, device=emissions.device)
        alpha = emissions[:, 0]
        for step in range(1, steps):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0)
            best, index = scores.max(dim=1)
            nxt = best + emissions[:, step]
            active = mask[:, step].unsqueeze(1)
            alpha = torch.where(active, nxt, alpha)
            back[:, step] = index
        lengths = mask.long().sum(dim=1).clamp(min=1)
        tags = torch.zeros(batch, steps, dtype=torch.long, device=emissions.device)
        chosen = alpha.argmax(dim=-1)
        for row in range(batch):
            pos = int(lengths[row]) - 1
            tags[row, pos] = chosen[row]
            for step in range(pos, 0, -1):
                chosen[row] = back[row, step, tags[row, step]]
                tags[row, step - 1] = chosen[row]
        return tags


def hidden_states(model, token_ids, pad_mask):
    """Contextual vector for every token, before the language-model head."""
    if isinstance(model, EncoderDecoderLM):
        return model.encode(token_ids, pad_mask)
    hidden = model.token_plus_position(token_ids)
    hidden = model.run_blocks(model.blocks, hidden, pad_mask)
    return model.norm(hidden)


class TokenTagger(nn.Module):
    def __init__(self, backbone, n_tags, tag_names=None, use_crf=False):
        super().__init__()
        self.backbone = backbone
        width = backbone.embed.weight.size(1)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(width, n_tags)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data[0] = 2.0
        self.crf = ChainCRF(tag_names) if use_crf else None

    def forward(self, token_ids, pad_mask):
        hidden = hidden_states(self.backbone, token_ids, pad_mask)
        return self.head(self.drop(hidden))


def build_tagger(arch, method, vocab_size, n_tags, weights_path, rank, alpha, tag_names=None, use_crf=False):
    backbone = build_model(arch, vocab_size=vocab_size)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state)
    if method == "full":
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        # NER uses encode() only. The causal decoder is unused, and its
        # gradients do not fit in 6GB next to a wider bidirectional window.
        if arch == "encdec":
            for parameter in backbone.decoder.parameters():
                parameter.requires_grad_(False)
    else:
        kind = "lorma" if method == "lorma" else "lora"
        wrapped = adapt_linears(backbone, kind, rank, alpha)
        print(f"wrapped {wrapped} linear layers with {kind}", flush=True)
    tagger = TokenTagger(backbone, n_tags, tag_names=tag_names, use_crf=use_crf)
    trainable = sum(parameter.numel() for parameter in tagger.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in tagger.parameters())
    print(f"parameters trainable {trainable / 1e6:.2f}M  total {total / 1e6:.2f}M", flush=True)
    return tagger


def class_weights(windows, n_tags, cap):
    counts = torch.ones(n_tags)
    for window in windows:
        for tag in window["target"]:
            counts[tag] += 1
    weights = 1.0 / counts.sqrt()
    weights = weights * (n_tags / weights.sum())
    weights = torch.clamp(weights, max=float(weights.median() * cap))
    return weights


def run_epoch(tagger, windows, optimizer, weights, device, train, batch_size, method, teacher=None, scaler=None):
    if train:
        tagger.train()
        # Frozen Phase 1 layers stay in eval so their dropout does not
        # scramble a 70-document training set. Adapter weights still learn.
        if method != "full":
            tagger.backbone.eval()
    else:
        tagger.eval()
    order = list(windows)
    if train:
        random.shuffle(order)
    total = 0.0
    seen = 0
    step = 0
    optimizer_was_used = optimizer is not None
    if optimizer_was_used:
        optimizer.zero_grad(set_to_none=True)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        width = max(len(item["ids"]) for item in chunk)
        token_ids = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
        targets = torch.full((len(chunk), width), -100, dtype=torch.long, device=device)
        pad_mask = torch.zeros(len(chunk), width, dtype=torch.bool, device=device)
        for row, item in enumerate(chunk):
            length = len(item["ids"])
            token_ids[row, :length] = torch.tensor(item["ids"], device=device)
            targets[row, :length] = torch.tensor(item["target"], device=device)
            pad_mask[row, :length] = True
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = tagger(token_ids, pad_mask).float()
                if tagger.crf is None:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        weight=weights,
                        ignore_index=-100,
                    )
                else:
                    with torch.autocast(device_type=device.type, enabled=False):
                        loss = tagger.crf.nll(logits.float(), targets, pad_mask)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(token_ids, pad_mask)
                    temperature = 2.0
                    student_log = F.log_softmax(logits / temperature, dim=-1)
                    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
                    kl = F.kl_div(student_log, teacher_prob, reduction="none").sum(dim=-1)
                    active = targets.ne(-100)
                    kl = (kl * active).sum() / active.sum().clamp(min=1)
                    loss = loss + (temperature * temperature) * kl
        if train:
            scaled = loss / 1.0
            if scaler is not None:
                scaler.scale(scaled).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                scaled.backward()
                torch.nn.utils.clip_grad_norm_(tagger.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach()) * len(chunk)
        seen += len(chunk)
        step += 1
        if train and step % 50 == 0:
            print(f"  step {step} loss {total / seen:.4f}", flush=True)
    return total / max(1, seen)


def trim_span(text, start, end):
    """Drop punctuation that is glued to a token but is outside the entity.

    The BPE often keeps the comma or sentence period on the last word.
    About 97% of the gold spans are an exact token span once those
    characters are removed. Periods that belong to 'v.' or 'Ltd.' stay.
    """
    while start < end and text[end - 1] in ",;:":
        end -= 1
    while start < end and text[start] == " ":
        start += 1
    if end - start >= 2 and text[end - 1] == "." and text[end - 2].isalnum():
        word = text[start:end]
        keep = word.endswith((" v.", " vs.", " No.", " Ltd.", " Pvt.", " Dr.", " Mr.", " Ms."))
        if not keep:
            end -= 1
    return start, end


def clean_edges(text, spans):
    """Fix citation edges that the tagger consistently gets one or two characters off.

    Held-out judgments keep the period on Ors., Anr. and Cr.P.C., and they
    leave out a glued footnote number, a closing parenthesis, and the words
    'reported in'. Party names are left alone: Ltd. is not a reliable period.
    """
    fixed = []
    for start, end, label in spans:
        if end > start and text[start:end].endswith(")."):
            end -= 2
        while end > start and text[end - 1] in ")]":
            end -= 1
        while end > start + 1 and text[end - 1].isdigit() and text[end - 2] in ".)":
            if end - start >= 5 and text[end - 5 : end - 1].isdigit():
                break
            end -= 1
        folded = text[start:end].casefold()
        marker = folded.find(" reported in ")
        if marker > 0:
            end = start + marker
        if label in ("PREC", "STAT", "CASENO") and end < len(text) and text[end] == ".":
            if text[start:end].endswith(("Ors", "Anr", "Cr.P.C", "I.P.C", "C.P.C")):
                end += 1
        if end > start:
            fixed.append((start, end, label))
    unique = []
    seen = set()
    for span in fixed:
        if span not in seen:
            seen.add(span)
            unique.append(span)
    return unique


def spans_from_tags(text, token_spans, tags, probabilities, thresholds):
    found = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if not tag.startswith("B-") and not tag.startswith("I-"):
            index += 1
            continue
        label = tag[2:]
        if probabilities[index] < thresholds.get(label, 0.0):
            index += 1
            continue
        cursor = index + 1
        while cursor < len(tags) and tags[cursor] == "I-" + label:
            if probabilities[cursor] < thresholds.get(label, 0.0):
                break
            cursor += 1
        start, end = trim_span(text, token_spans[index][0], token_spans[cursor - 1][1])
        if end > start:
            found.append((start, end, label))
        index = cursor
    # The evaluator rejects a repeated span, so keep the first copy only.
    unique = []
    seen = set()
    for span in found:
        if span not in seen:
            seen.add(span)
            unique.append(span)
    return unique


def propagate_spans(text, spans):
    """Copy a name the model already found onto later exact mentions.

    About two thirds of the gold appellant and respondent spans are the same
    string repeated in one judgment. A window tagger often hits the name once
    and misses the copies. Only labels that are names are copied, and a copy
    that sits inside a span already predicted is left alone.
    """
    occupied = [(start, end) for start, end, _label in spans]
    added = []

    def overlaps(start, end):
        return any(not (end <= left or start >= right) for left, right in occupied)

    def boundary(index):
        return index <= 0 or index >= len(text) or not text[index].isalnum()

    # Respondent, witness, and counsel copies added more wrong spans than
    # right ones on the held-out judgments. Appellant names did the opposite.
    minimum = {"APP": 8}
    folded = text.casefold()
    seen = set()
    for start, end, label in spans:
        if label not in minimum:
            continue
        raw = " ".join(text[start:end].split())
        key = raw.casefold()
        if len(key) < minimum[label] or key in _GENERIC or (label, key) in seen:
            continue
        seen.add((label, key))
        needle = key
        cursor = 0
        while True:
            at = folded.find(needle, cursor)
            if at < 0:
                break
            stop = at + len(needle)
            cursor = at + 1
            if not boundary(at - 1) or not boundary(stop):
                continue
            if overlaps(at, stop):
                continue
            tail = folded[stop : stop + 8]
            if label in ("APP", "RESP") and (
                tail.startswith(" v.") or tail.startswith(" vs") or tail.startswith(" versus")
            ):
                continue
            added.append((at, stop, label))
            occupied.append((at, stop))
    return spans + added


def predict_document(tagger, tokenizer, text, width, stride, tag_names, thresholds, device):
    ids, spans = tokenizer.encode_with_offsets(text)
    if not ids:
        return []
    summed = torch.zeros(len(ids), len(tag_names))
    counts = torch.zeros(len(ids), 1)
    tagger.eval()
    for window in make_windows(ids, spans, ["O"] * len(ids), width, stride):
        token_ids = torch.tensor([window["ids"]], dtype=torch.long, device=device)
        pad_mask = torch.ones(1, len(window["ids"]), dtype=torch.bool, device=device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = tagger(token_ids, pad_mask)[0].float().cpu()
        begin = window["offset"]
        summed[begin : begin + len(window["ids"])] += logits
        counts[begin : begin + len(window["ids"])] += 1
    averaged = summed / counts.clamp(min=1)
    if tagger.crf is None:
        probabilities, chosen = averaged.softmax(dim=-1).max(dim=-1)
        chosen = chosen.tolist()
        probabilities = probabilities.tolist()
    else:
        device = next(tagger.parameters()).device
        emissions = averaged.unsqueeze(0).to(device)
        mask = torch.ones(1, averaged.size(0), dtype=torch.bool, device=device)
        chosen = tagger.crf.decode(emissions, mask)[0].cpu().tolist()
        probabilities = averaged.softmax(dim=-1).gather(
            1, torch.tensor(chosen).unsqueeze(1)
        ).squeeze(1).tolist()
    tags = [tag_names[index] for index in chosen]
    found = spans_from_tags(text, spans, tags, probabilities, thresholds)
    if PROPAGATE:
        found = propagate_spans(text, found)
    return clean_edges(text, found)


def macro_f1(gold_rows, predictions):
    stats = {label: [0, 0, 0] for label in LABELS}
    for doc_id, _text, entities in gold_rows:
        gold = {(start, end, label) for start, end, label in entities}
        pred = set(tuple(span) for span in predictions.get(doc_id, []))
        for label in LABELS:
            gold_label = {span for span in gold if span[2] == label}
            pred_label = {span for span in pred if span[2] == label}
            stats[label][0] += len(gold_label & pred_label)
            stats[label][1] += len(pred_label - gold_label)
            stats[label][2] += len(gold_label - pred_label)
    f1s = []
    lines = []
    for label in LABELS:
        true_pos, false_pos, false_neg = stats[label]
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        lines.append(f"{label}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    return sum(f1s) / len(f1s), lines


def score_split(tagger, tokenizer, rows, width, stride, tag_names, thresholds, device):
    predictions = {}
    for index, (doc_id, text, _entities) in enumerate(rows):
        predictions[doc_id] = predict_document(
            tagger, tokenizer, text, width, stride, tag_names, thresholds, device
        )
        if index % 5 == 0:
            print(f"  scored {index + 1}/{len(rows)}", flush=True)
    f1, lines = macro_f1(rows, predictions)
    return f1, lines, predictions


def tune_thresholds(tagger, tokenizer, rows, width, stride, tag_names, device):
    """Pick one minimum probability per label on the held-out documents."""
    _f1, _lines, predictions = score_split(
        tagger, tokenizer, rows, width, stride, tag_names, {label: 0.0 for label in LABELS}, device
    )
    # Re-score from cached spans is not available, so search a short grid by
    # re-decoding is too expensive. The probabilities are already inside
    # predict_document. A second pass with a shared threshold is the stable
    # choice on only a handful of dev documents.
    best_f1 = _f1
    best = {label: 0.0 for label in LABELS}
    print(f"threshold 0.00  macro-F1 {best_f1:.4f}", flush=True)
    for minimum in (0.15, 0.30, 0.45, 0.60, 0.75):
        trial = {label: minimum for label in LABELS}
        f1, _lines, _predictions = score_split(
            tagger, tokenizer, rows, width, stride, tag_names, trial, device
        )
        print(f"threshold {minimum:.2f}  macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best = trial
    return best, best_f1


def split_documents(fold1, fold2, dev_count):
    ordered = sorted(fold2, key=lambda row: row[0])
    dev_rows = ordered[-dev_count:]
    dev_ids = {row[0] for row in dev_rows}
    train_rows = list(fold1) + [row for row in fold2 if row[0] not in dev_ids]
    return train_rows, dev_rows


def collect_windows(tokenizer, rows, width, stride, tag_index):
    windows = []
    for doc_id, text, entities in rows:
        _ids, _spans, _tags, doc_windows = prepare_document(
            tokenizer, doc_id, text, entities, width, stride, tag_index
        )
        windows.extend(doc_windows)
    return windows


def save_checkpoint(path, tagger, args, tag_names, thresholds, dev_f1):
    torch.save(
        {
            "arch": args.arch,
            "method": args.method,
            "tag_names": tag_names,
            "thresholds": thresholds,
            "dev_f1": dev_f1,
            "model": tagger.state_dict(),
        },
        path,
    )
    print("saved", path, flush=True)


def write_submission(path, predictions):
    payload = {doc_id: [list(span) for span in spans] for doc_id, spans in predictions.items()}
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    pred_path = os.path.join(folder, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pred_path, arcname="prediction.json")
    print("zip", path, "docs", len(payload), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["decoder", "encdec", "hybrid", "efficient"])
    parser.add_argument("--method", required=True, choices=["full", "lora", "lorma", "distill"])
    parser.add_argument("--weights", default=None)
    parser.add_argument("--bpe", default=DEFAULT_BPE)
    parser.add_argument("--teacher", default=None, help="Checkpoint from a larger Phase 2 model.")
    parser.add_argument("--teacher_arch", default="efficient")
    parser.add_argument("--teacher_method", default="lorma")
    parser.add_argument("--teacher_weights", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--head_lr", type=float, default=0.0, help="Learning rate for the tag head and CRF.")
    parser.add_argument("--dev_count", type=int, default=8)
    parser.add_argument("--weight_cap", type=float, default=5.0)
    parser.add_argument("--crf", action="store_true")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--data_dir", default=None, help="Local Phase_2_LNER folder, skipping Hugging Face.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", default=None, help="Phase 2 checkpoint to continue from.")
    parser.add_argument("--unfreeze", action="store_true", help="Train the Phase 1 weights under the adapter.")
    parser.add_argument("--score_only", action="store_true", help="Score the resume checkpoint and write a zip.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    global DATA_DIR
    DATA_DIR = args.data_dir

    if args.method == "full":
        args.batch = args.batch or 1
        args.width = args.width or 256
        args.stride = args.stride or 128
        args.lr = args.lr or 2e-5
    else:
        args.batch = args.batch or 2
        args.width = args.width or 384
        args.stride = args.stride or 192
        args.lr = args.lr or 1e-4
    if args.weights is None:
        args.weights = DEFAULT_WEIGHTS[args.arch]
    if args.out is None:
        args.out = os.path.join(LOCAL, "phase2_runs", f"{args.arch}_{args.method}")
    os.makedirs(args.out, exist_ok=True)

    set_seed(args.seed)
    print("seed", args.seed, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    if not os.path.isfile(args.weights):
        raise SystemExit(f"missing Phase 1 weights: {args.weights}")
    if not os.path.isfile(args.bpe):
        raise SystemExit(f"missing BPE: {args.bpe}")

    tokenizer = LegalBPE.load(args.bpe)
    fold1 = load_fold("Phase_2_LNER/Train/fold1-text.json", "Phase_2_LNER/Train/fold1-gold.json")
    fold2 = load_fold("Phase_2_LNER/Train/fold2-text.json", "Phase_2_LNER/Train/fold2-gold.json")
    print("fold1", len(fold1), "fold2", len(fold2), flush=True)
    covered, total = boundary_report(tokenizer, fold1 + fold2)
    if total:
        print(f"exact token-boundary rate {covered / total:.3f}", flush=True)
    if args.check:
        return

    train_rows, dev_rows = split_documents(fold1, fold2, args.dev_count)
    print("train docs", len(train_rows), "dev docs", [row[0] for row in dev_rows], flush=True)
    names = tag_names_of()
    tag_index = {name: index for index, name in enumerate(names)}
    train_windows = collect_windows(tokenizer, train_rows, args.width, args.stride, tag_index)
    print("train windows", len(train_windows), "width", args.width, "stride", args.stride, flush=True)

    tagger = build_tagger(
        args.arch, args.method, len(tokenizer), len(names), args.weights, args.rank, args.alpha, names, args.crf
    )
    if args.bidirectional:
        opened = enable_full_attention(tagger.backbone)
        print("full attention layers", opened, flush=True)
    best_path = os.path.join(args.out, "best.pt")
    best_f1 = -1.0
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        tagger.load_state_dict(resumed["model"])
        best_f1 = float(resumed.get("dev_f1", -1.0))
        print("resumed", args.resume, "dev macro-F1", round(best_f1, 4), flush=True)
    if args.unfreeze:
        opened = 0
        for module in tagger.modules():
            if isinstance(module, (LoRMALinear, LoRALinear)):
                module.base.weight.requires_grad_(True)
                if module.base.bias is not None:
                    module.base.bias.requires_grad_(True)
                opened += 1
        print("unfroze", opened, "adapter base matrices", flush=True)
    tagger = tagger.to(device)
    teacher = None
    if args.method == "distill":
        if not args.teacher:
            raise SystemExit("distill needs --teacher pointing at a Phase 2 checkpoint")
        teacher_weights = args.teacher_weights or DEFAULT_WEIGHTS[args.teacher_arch]
        checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=True)
        teacher_state = checkpoint["model"]
        teacher_crf = any(key.startswith("crf.") for key in teacher_state)
        teacher = build_tagger(
            args.teacher_arch,
            args.teacher_method,
            len(tokenizer),
            len(names),
            teacher_weights,
            args.rank,
            args.alpha,
            names,
            teacher_crf,
        )
        teacher.load_state_dict(teacher_state)
        if args.bidirectional:
            opened = enable_full_attention(teacher.backbone)
            print("teacher full attention layers", opened, flush=True)
        teacher = teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    weights = class_weights(train_windows, len(names), args.weight_cap).to(device)
    if tagger.crf is not None:
        tagger.crf.score_bias.copy_(weights.log().to(tagger.crf.score_bias.device))
    if args.score_only:
        global PROPAGATE
        zero = {label: 0.0 for label in LABELS}
        PROPAGATE = False
        base_f1, base_lines, _base_pred = score_split(
            tagger, tokenizer, dev_rows, args.width, args.stride, names, zero, device
        )
        print(f"base dev macro-F1 {base_f1:.4f}", flush=True)
        print("\n".join(base_lines), flush=True)
        PROPAGATE = True
        prop_f1, prop_lines, _prop_pred = score_split(
            tagger, tokenizer, dev_rows, args.width, args.stride, names, zero, device
        )
        print(f"propagate dev macro-F1 {prop_f1:.4f}", flush=True)
        print("\n".join(prop_lines), flush=True)
        PROPAGATE = prop_f1 > base_f1 + 1e-4
        print("zip uses propagate" if PROPAGATE else "zip uses the model spans only", flush=True)
        test_rows = load_fold("Phase_2_LNER/Test/fold3-text.json")
        predictions = {}
        for index, (doc_id, text, _entities) in enumerate(test_rows):
            predictions[doc_id] = predict_document(
                tagger, tokenizer, text, args.width, args.stride, names, zero, device
            )
            print(f"test {index + 1}/{len(test_rows)} id {doc_id} spans {len(predictions[doc_id])}", flush=True)
        write_submission(os.path.join(args.out, "submission.zip"), predictions)
        return
    head_params = []
    backbone_params = []
    for name, parameter in tagger.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head.") or name.startswith("crf."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    if args.head_lr:
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr},
                {"params": head_params, "lr": args.head_lr},
            ],
            weight_decay=0.01,
        )
        print(f"lr backbone {args.lr} head {args.head_lr}", flush=True)
    else:
        optimizer = torch.optim.AdamW(backbone_params + head_params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if args.resume and not os.path.isfile(best_path):
        shutil.copyfile(args.resume, best_path)

    patience = 0
    for epoch in range(args.epochs):
        loss = run_epoch(
            tagger, train_windows, optimizer, weights, device, True, args.batch, args.method, teacher, scaler
        )
        print(f"epoch {epoch + 1} train loss {loss:.4f}", flush=True)
        f1, lines, _predictions = score_split(
            tagger, tokenizer, dev_rows, args.width, args.stride, names, {label: 0.0 for label in LABELS}, device
        )
        print(f"epoch {epoch + 1} dev macro-F1 {f1:.4f}", flush=True)
        print("\n".join(lines), flush=True)
        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            save_checkpoint(best_path, tagger, args, names, {label: 0.0 for label in LABELS}, f1)
        else:
            patience += 1
            if patience >= 3:
                print("early stop", flush=True)
                break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    tagger.load_state_dict(checkpoint["model"])
    tagger.to(device)
    thresholds, tuned_f1 = tune_thresholds(
        tagger, tokenizer, dev_rows, args.width, args.stride, names, device
    )
    print("chosen thresholds", thresholds, "dev macro-F1", round(tuned_f1, 4), flush=True)
    save_checkpoint(best_path, tagger, args, names, thresholds, tuned_f1)

    test_rows = load_fold("Phase_2_LNER/Test/fold3-text.json")
    predictions = {}
    for index, (doc_id, text, _entities) in enumerate(test_rows):
        predictions[doc_id] = predict_document(
            tagger, tokenizer, text, args.width, args.stride, names, thresholds, device
        )
        print(f"test {index + 1}/{len(test_rows)} id {doc_id} spans {len(predictions[doc_id])}", flush=True)
    write_submission(os.path.join(args.out, "submission.zip"), predictions)


if __name__ == "__main__":
    main()
