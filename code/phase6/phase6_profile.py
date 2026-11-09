"""Print Phase 6 train/test size stats. Uses the Phase 1 BPE on a sample."""
import json
import os
import random
import statistics

from legal_bpe import LegalBPE
from phase4_own import DEFAULT_BPE, LOCAL

ROOT = os.path.join(LOCAL, "phase6data", "Phase_6_Summ")


def load(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        first = next(iter(payload.values()))
        if isinstance(first, dict):
            return list(payload.values())
        return [{"id": key, "document": value, "summary": ""} for key, value in payload.items()]
    raise SystemExit(type(payload))


def main():
    train = load(os.path.join(ROOT, "Train", "train.json"))
    test = load(os.path.join(ROOT, "Test", "test.json"))
    print("n_train", len(train), "n_test", len(test))
    print("train keys", sorted(train[0].keys()))
    print("test keys", sorted(test[0].keys()))
    print("train id", train[0].get("id"), "test id", test[0].get("id"))
    doc_words = [len(str(row.get("document", "")).split()) for row in train]
    sum_words = [len(str(row.get("summary", "")).split()) for row in train]
    test_words = [len(str(row.get("document", row if not isinstance(row, dict) else "")).split()) for row in test]

    def show(name, values):
        values = sorted(values)
        print(
            name,
            "mean", round(statistics.mean(values), 1),
            "p50", values[len(values) // 2],
            "p90", values[int(len(values) * 0.9)],
            "p99", values[int(len(values) * 0.99)],
            "max", values[-1],
        )

    show("train_doc_words", doc_words)
    show("train_sum_words", sum_words)
    show("test_doc_words", test_words)
    print("sample summary:", str(train[0].get("summary", ""))[:240])
    tokenizer = LegalBPE.load(DEFAULT_BPE)
    sample = random.Random(0).sample(train, min(400, len(train)))
    src_tok = [len(tokenizer.encode(row["document"])) for row in sample]
    tgt_tok = [len(tokenizer.encode(row["summary"])) for row in sample]
    show("sample_src_tok", src_tok)
    show("sample_tgt_tok", tgt_tok)
    over = sum(1 for n in src_tok if n > 1024) / len(src_tok)
    print("frac src > 1024", round(over, 3), "frac src > 768", round(sum(1 for n in src_tok if n > 768) / len(src_tok), 3))


if __name__ == "__main__":
    main()
