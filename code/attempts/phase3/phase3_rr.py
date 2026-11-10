import os
import json
import random
import zipfile
from collections import Counter

import torch
import torch.nn as nn
from huggingface_hub import login, hf_hub_download
from kaggle_secrets import UserSecretsClient

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
HF_TOKEN = UserSecretsClient().get_secret("HF")
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)
REPO = "Exploration-Lab/CS-781-Capstone"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = "/kaggle/working/cs781_phase3"
os.makedirs(OUT, exist_ok=True)
print("device", DEVICE)


def fetch(name):
    path = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset", token=HF_TOKEN)
    print("got", name)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first in "[{":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def sentence_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "sentence", "content"):
            if isinstance(item.get(key), str):
                return item[key]
    return ""


def label_text(item, vocab):
    if isinstance(item, str) and item in vocab:
        return item
    if isinstance(item, dict):
        for key in ("label", "role", "labels"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return None


def docs_from(payload, vocab):
    records = []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if any(key in payload for key in ("annotations", "sentences", "text")):
            records = [payload]
        else:
            for key, value in payload.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("id", str(key))
                    records.append(item)
                else:
                    records.append({"id": str(key), "value": value})
    docs = []
    for record in records:
        doc_id = str(record.get("id") or record.get("doc_id") or len(docs))
        sentences = None
        labels = None
        if isinstance(record.get("sentences"), list):
            sentences = [sentence_text(item) for item in record["sentences"]]
            if isinstance(record.get("labels"), list):
                labels = [str(item) for item in record["labels"]]
        elif isinstance(record.get("annotations"), list):
            sentences = []
            labels = []
            for item in record["annotations"]:
                sentences.append(sentence_text(item))
                found = label_text(item, vocab)
                if found is not None:
                    labels.append(found)
            if len(labels) != len(sentences):
                labels = None
        elif isinstance(record.get("value"), list) and record["value"]:
            value = record["value"]
            if isinstance(value[0], str):
                if all(item in vocab for item in value):
                    labels = [str(item) for item in value]
                    sentences = [""] * len(labels)
                else:
                    sentences = [str(item) for item in value]
            elif isinstance(value[0], dict):
                sentences = [sentence_text(item) for item in value]
                labels = []
                for item in value:
                    found = label_text(item, vocab)
                    if found is not None:
                        labels.append(found)
                if len(labels) != len(sentences):
                    labels = None
        if sentences is None:
            continue
        docs.append({"id": doc_id, "sentences": sentences, "labels": labels})
    return docs


vocab_payload = read_json(fetch("Phase_3_RR/Train/label_vocab.json"))
if isinstance(vocab_payload, list):
    VOCAB = [str(item) for item in vocab_payload]
elif isinstance(vocab_payload, dict):
    values = list(vocab_payload.values())
    if values and all(isinstance(item, str) for item in values):
        pairs = sorted(vocab_payload.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0])
        VOCAB = [str(item[1]) for item in pairs]
    else:
        VOCAB = [str(item) for item in vocab_payload.keys()]
else:
    raise SystemExit("label_vocab.json has an unexpected shape")
print("vocab", VOCAB)
if len(VOCAB) != 13:
    raise SystemExit(f"expected 13 labels, found {len(VOCAB)}")
label_index = {name: index for index, name in enumerate(VOCAB)}
vocab_set = set(VOCAB)

train_docs = []
for name in (
    "Phase_3_RR/Train/CL_train.json",
    "Phase_3_RR/Train/IT_train.json",
):
    found = docs_from(read_json(fetch(name)), vocab_set)
    print(name, "docs", len(found))
    train_docs.extend(found)

dev_docs = []
for name in (
    "Phase_3_RR/Train/CL_dev.json",
    "Phase_3_RR/Train/IT_dev.json",
):
    found = docs_from(read_json(fetch(name)), vocab_set)
    print(name, "docs", len(found))
    dev_docs.extend(found)

train_docs = [doc for doc in train_docs if doc["labels"] and len(doc["labels"]) == len(doc["sentences"])]
dev_docs = [doc for doc in dev_docs if doc["labels"] and len(doc["labels"]) == len(doc["sentences"])]
print("usable train docs", len(train_docs), "dev docs", len(dev_docs))
if not train_docs:
    sample = read_json(fetch("Phase_3_RR/Train/CL_train.json"))
    if isinstance(sample, dict):
        key = next(iter(sample))
        print("sample key", key, type(sample[key]).__name__)
        print(str(sample[key])[:500])
    elif isinstance(sample, list) and sample:
        print("sample item", str(sample[0])[:500])
    raise SystemExit("could not read sentence labels from the train files")

counts = Counter()
for doc in train_docs:
    for sentence in doc["sentences"]:
        for word in sentence.split():
            counts[word] += 1
itos = ["<pad>", "<unk>"] + [word for word, count in counts.most_common(25000) if count >= 2]
stoi = {word: index for index, word in enumerate(itos)}


def encode_sentence(sentence):
    ids = [stoi.get(word, 1) for word in sentence.split()[:80]]
    return ids or [1]


class SentenceTagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(itos), 128, padding_idx=0)
        self.out = nn.Linear(128, len(VOCAB))

    def forward(self, token_ids, mask):
        x = self.embed(token_ids)
        mask = mask.unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.out(pooled)


rows = []
for doc in train_docs:
    for sentence, label in zip(doc["sentences"], doc["labels"]):
        if label not in label_index:
            continue
        rows.append((encode_sentence(sentence), label_index[label]))
print("train sentences", len(rows))

model = SentenceTagger().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()


def batches(data, size, shuffle):
    order = list(data)
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), size):
        chunk = order[start : start + size]
        width = max(len(item[0]) for item in chunk)
        token_ids = torch.zeros(len(chunk), width, dtype=torch.long, device=DEVICE)
        mask = torch.zeros(len(chunk), width, dtype=torch.float, device=DEVICE)
        target = torch.tensor([item[1] for item in chunk], dtype=torch.long, device=DEVICE)
        for row, (ids, _label) in enumerate(chunk):
            token_ids[row, : len(ids)] = torch.tensor(ids, device=DEVICE)
            mask[row, : len(ids)] = 1
        yield token_ids, mask, target


for epoch in range(3):
    model.train()
    total = 0.0
    seen = 0
    for token_ids, mask, target in batches(rows, 64, True):
        logits = model(token_ids, mask)
        loss = loss_fn(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss) * len(target)
        seen += len(target)
    print(f"epoch {epoch + 1} loss {total / max(1, seen):.4f}")


def predict_sentences(sentences):
    model.eval()
    labels = []
    encoded = [(encode_sentence(sentence), 0) for sentence in sentences]
    with torch.no_grad():
        for token_ids, mask, _target in batches(encoded, 64, False):
            pred = model(token_ids, mask).argmax(dim=-1).tolist()
            labels.extend(VOCAB[index] for index in pred)
    return labels


correct = 0
total = 0
for doc in dev_docs:
    pred = predict_sentences(doc["sentences"])
    for gold, guess in zip(doc["labels"], pred):
        correct += int(gold == guess)
        total += 1
print(f"dev sentence accuracy {correct / max(1, total):.4f} on {total}")

predictions = {}
for name in (
    "Phase_3_RR/Test/CL_test.json",
    "Phase_3_RR/Test/IT_test.json",
):
    docs = docs_from(read_json(fetch(name)), vocab_set)
    print(name, "docs", len(docs))
    for doc in docs:
        doc_id = doc["id"][:-4] if doc["id"].endswith(".txt") else doc["id"]
        predictions[doc_id] = predict_sentences(doc["sentences"])

pred_path = os.path.join(OUT, "prediction.json")
zip_path = os.path.join(OUT, "submission.zip")
with open(pred_path, "w", encoding="utf-8") as handle:
    json.dump(predictions, handle, ensure_ascii=False)
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    bundle.write(pred_path, arcname="prediction.json")
print("zip", zip_path, "docs", len(predictions))
