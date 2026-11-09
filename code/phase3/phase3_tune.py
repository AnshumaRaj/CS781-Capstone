"""Average cached Phase 3 role scores and tune one score offset per role for macro-F1.

    python phase3_tune.py run_a run_b ... [--out folder --write] [--shrink 0.5]

Each run folder needs scores.pt from phase3_cache.py. The offsets are added to the
emissions before CRF decoding. Roles with no dev sentence keep offset 0. A cross
check tunes on the CL dev documents and scores the IT ones, and the other way round.
"""
import argparse
import os

import numpy as np
import torch

import phase3_rr_own as rr


def viterbi(emissions, start, end, trans):
    score = start + emissions[0]
    back = np.empty((len(emissions), len(start)), dtype=np.int64)
    for step in range(1, len(emissions)):
        total = score[:, None] + trans
        back[step] = total.argmax(0)
        score = total.max(0) + emissions[step]
    path = [int((score + end).argmax())]
    for step in range(len(emissions) - 1, 0, -1):
        path.append(int(back[step, path[-1]]))
    return path[::-1]


def macro(gold, pred, n_labels):
    f1 = np.zeros(n_labels)
    for label in range(n_labels):
        tp = np.sum((gold == label) & (pred == label))
        fp = np.sum((gold != label) & (pred == label))
        fn = np.sum((gold == label) & (pred != label))
        if tp:
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            f1[label] = 2 * precision * recall / (precision + recall)
    return f1.mean()


class Scorer:
    def __init__(self, docs, scores, crf, vocab):
        self.docs = docs
        self.scores = scores
        self.start, self.end, self.trans = crf["start"], crf["end"], crf["trans"]
        self.index = {label: i for i, label in enumerate(vocab)}
        self.n = len(vocab)

    def decode(self, doc_ids, bias):
        return {doc_id: viterbi(self.scores[doc_id] + bias, self.start, self.end, self.trans) for doc_id in doc_ids}

    def f1(self, docs, bias):
        paths = self.decode([doc["id"] for doc in docs], bias)
        gold = np.concatenate([[self.index[label] for label in doc["labels"]] for doc in docs])
        pred = np.concatenate([paths[doc["id"]] for doc in docs])
        return macro(gold, pred, self.n)


def tune(scorer, docs, free, grid, passes):
    bias = np.zeros(scorer.n)
    best = scorer.f1(docs, bias)
    for _ in range(passes):
        moved = False
        for label in free:
            keep = bias[label]
            for value in grid:
                bias[label] = value
                f1 = scorer.f1(docs, bias)
                if f1 > best + 1e-9:
                    best, keep, moved = f1, value, True
            bias[label] = keep
        if not moved:
            break
    return bias, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--data_dir", default=rr.DEFAULT_DATA)
    parser.add_argument("--out")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--shrink", type=float, default=1.0, help="Scale the final offsets towards 0.")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--no_tune", action="store_true", help="Plain average, no offsets.")
    parser.add_argument("--rarity", type=float, help="Add this times the centred -log train frequency of each role.")
    parser.add_argument("--rarity_scan", action="store_true")
    parser.add_argument("--trans_scale", type=float, default=1.0, help="Multiply the CRF transition scores.")
    args = parser.parse_args()

    vocab = rr.read_vocab(args.data_dir)
    dev_docs = rr.load_docs(args.data_dir, rr.DEV_FILES)
    test_docs = rr.load_docs(args.data_dir, rr.TEST_FILES)

    caches = [torch.load(os.path.join(run, "scores.pt"), map_location="cpu", weights_only=False) for run in args.runs]
    scores = {doc_id: np.mean([c["scores"][doc_id].numpy() for c in caches], axis=0) for doc_id in caches[0]["scores"]}
    crf = {name: np.mean([c["crf"][name].numpy() for c in caches], axis=0) for name in caches[0]["crf"]}
    crf["trans"] = crf["trans"] * args.trans_scale
    scorer = Scorer(dev_docs, scores, crf, vocab)
    for run, cache in zip(args.runs, caches):
        print(f"{os.path.basename(run):28s} stored dev {cache['dev_f1']:.4f}", flush=True)

    present = {label for doc in dev_docs for label in doc["labels"]}
    free = [scorer.index[label] for label in vocab if label in present]
    spread = float(np.std(np.concatenate([scores[doc["id"]] for doc in dev_docs])))
    grid = spread * np.arange(-1.0, 2.01, 0.125)

    zero = np.zeros(len(vocab))
    base = scorer.f1(dev_docs, zero)
    print(f"averaged {len(caches)}  dev {base:.4f}  emission spread {spread:.3f}", flush=True)
    if args.no_tune:
        args.shrink, args.passes = 0.0, 0

    prior_bias = np.zeros(len(vocab))
    if args.rarity is not None or args.rarity_scan:
        train_docs = rr.load_docs(args.data_dir, rr.TRAIN_FILES)
        counts = np.ones(len(vocab))
        for doc in train_docs:
            for label in doc["labels"]:
                counts[scorer.index[label]] += 1
        rarity = -np.log(counts / counts.sum())
        rarity -= rarity.mean()
        cl = [doc for doc in dev_docs if doc["domain"] == "CL"]
        it = [doc for doc in dev_docs if doc["domain"] == "IT"]
        if args.rarity_scan:
            for tau in np.arange(0.0, 1.01, 0.1):
                bias = tau * rarity
                print(
                    f"rarity {tau:.1f}  dev {scorer.f1(dev_docs, bias):.4f}  CL {scorer.f1(cl, bias):.4f}  IT {scorer.f1(it, bias):.4f}",
                    flush=True,
                )
        prior_bias = (args.rarity or 0.0) * rarity
        print(f"chosen rarity {args.rarity}  dev {scorer.f1(dev_docs, prior_bias):.4f}", flush=True)
        args.passes, args.shrink = 0, 0.0
        args.no_tune = True

    cl = [doc for doc in dev_docs if doc["domain"] == "CL"]
    it = [doc for doc in dev_docs if doc["domain"] == "IT"]
    for fit, held, name in ((cl, it, "CL->IT"), (it, cl, "IT->CL")) if not args.no_tune else ():
        bias, _ = tune(scorer, fit, free, grid, args.passes)
        before, after, half = scorer.f1(held, zero), scorer.f1(held, bias), scorer.f1(held, 0.5 * bias)
        print(f"cross {name}  held-out {before:.4f} -> full offsets {after:.4f}  half offsets {half:.4f}", flush=True)

    bias, tuned = tune(scorer, dev_docs, free, grid, args.passes)
    final = args.shrink * bias + prior_bias
    print(f"tuned on all dev {tuned:.4f}  with shrink {args.shrink}: {scorer.f1(dev_docs, final):.4f}", flush=True)
    print("offsets", {vocab[i]: round(float(final[i]), 3) for i in range(len(vocab)) if final[i]}, flush=True)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        dev_pred = {k: [vocab[i] for i in v] for k, v in scorer.decode([d["id"] for d in dev_docs], final).items()}
        rr.write_dev_files(args.out, dev_docs, dev_pred)
        if args.write:
            test_pred = {k: [vocab[i] for i in v] for k, v in scorer.decode([d["id"] for d in test_docs], final).items()}
            for doc in test_docs:
                if len(test_pred[doc["id"]]) != len(doc["text"]):
                    raise SystemExit(f"{doc['id']}: wrong number of roles")
            rr.write_outputs(args.out, test_pred)
            print("zip", os.path.join(args.out, "submission.zip"), flush=True)


if __name__ == "__main__":
    main()
