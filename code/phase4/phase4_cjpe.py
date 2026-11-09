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
OUT = "/kaggle/working/cs781_phase4"
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


def rows_from(payload):
    rows = []
    if isinstance(payload, list):
        source = payload
    elif isinstance(payload, dict):
        if "text" in payload or "label" in payload or "gold" in payload:
            source = [payload]
        else:
            source = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("id", str(key))
                    source.append(item)
                elif isinstance(value, str):
                    source.append({"id": str(key), "text": value})
                else:
                    source.append({"id": str(key), "label": value})
    else:
        source = []
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("judgment") or item.get("content") or ""
        label = item.get("label", item.get("gold", item.get("y")))
        doc_id = str(item.get("id") or item.get("doc_id") or index)
        rows.append((doc_id, str(text), label))
    return rows


def as_bit(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    return None


train_rows = rows_from(read_json(fetch("Phase_4_CJPE/Train/train.json")))
dev_rows = rows_from(read_json(fetch("Phase_4_CJPE/Train/dev.json")))
test_rows = rows_from(read_json(fetch("Phase_4_CJPE/Test/test.json")))
print("train", len(train_rows), "dev", len(dev_rows), "test", len(test_rows))
train_rows = [(doc_id, text, as_bit(label)) for doc_id, text, label in train_rows]
train_rows = [row for row in train_rows if row[2] is not None and row[1].strip()]
dev_labeled = []
for doc_id, text, label in dev_rows:
    bit = as_bit(label)
    if bit is not None and text.strip():
        dev_labeled.append((doc_id, text, bit))
print("usable train", len(train_rows), "usable dev", len(dev_labeled))
if not train_rows:
    raise SystemExit("could not read id/text/label from train.json")

counts = Counter()
for _doc_id, text, _label in train_rows:
    for word in text.split()[:256]:
        counts[word] += 1
itos = ["<pad>", "<unk>"] + [word for word, count in counts.most_common(40000) if count >= 3]
stoi = {word: index for index, word in enumerate(itos)}
print("vocab", len(itos))


def encode(text):
    ids = [stoi.get(word, 1) for word in text.split()[:256]]
    return ids or [1]


class DocClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(itos), 128, padding_idx=0)
        self.out = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, token_ids, mask):
        x = self.embed(token_ids)
        mask = mask.unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.out(pooled)


train_encoded = [(encode(text), label) for _doc_id, text, label in train_rows]
dev_encoded = [(encode(text), label) for _doc_id, text, label in dev_labeled]
del train_rows
model = DocClassifier().to(DEVICE)
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


for epoch in range(2):
    model.train()
    total = 0.0
    seen = 0
    for token_ids, mask, target in batches(train_encoded, 32, True):
        loss = loss_fn(model(token_ids, mask), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss) * len(target)
        seen += len(target)
    print(f"epoch {epoch + 1} loss {total / max(1, seen):.4f}")


def predict_encoded(encoded_rows):
    model.eval()
    guesses = []
    with torch.no_grad():
        for token_ids, mask, _target in batches(encoded_rows, 32, False):
            guesses.extend(model(token_ids, mask).argmax(dim=-1).tolist())
    return guesses


if dev_encoded:
    guesses = predict_encoded(dev_encoded)
    gold = [label for _ids, label in dev_encoded]
    correct = sum(int(a == b) for a, b in zip(gold, guesses))
    print(f"dev accuracy {correct / len(gold):.4f}")

test_encoded = [(encode(text), 0) for _doc_id, text, _label in test_rows]
guesses = predict_encoded(test_encoded)
predictions = []
for (doc_id, _text, _label), guess in zip(test_rows, guesses):
    predictions.append({"id": doc_id, "pred": int(guess)})

pred_path = os.path.join(OUT, "prediction.json")
zip_path = os.path.join(OUT, "submission.zip")
with open(pred_path, "w", encoding="utf-8") as handle:
    json.dump(predictions, handle, ensure_ascii=False)
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    bundle.write(pred_path, arcname="prediction.json")
print("zip", zip_path, "rows", len(predictions))
