"""Average the role scores of several finished Phase 3 runs.

Each checkpoint is loaded in turn, so only one Phase 1 model sits on the GPU.
The per-sentence role scores are averaged, the CRF parameters are averaged,
and the CRF picks one role per sentence from the averaged scores.

    python phase3_ensemble.py --out <folder> run_a/best.pt run_b/best.pt ...
"""
import argparse
import os

import torch

import phase3_rr_own as rr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bpe", default=rr.DEFAULT_BPE)
    parser.add_argument("--data_dir", default=rr.DEFAULT_DATA)
    parser.add_argument("--eval_tokens", type=int, default=160)
    parser.add_argument("--write", action="store_true", help="Write submission.zip for the test documents.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = rr.read_vocab(args.data_dir)
    dev_docs = rr.load_docs(args.data_dir, rr.DEV_FILES)
    test_docs = rr.load_docs(args.data_dir, rr.TEST_FILES)
    tokenizer = rr.LegalBPE.load(args.bpe)
    rr.tokenize_docs(tokenizer, dev_docs + test_docs, args.eval_tokens)
    docs = dev_docs + test_docs if args.write else dev_docs

    summed = {}
    crf_sum = None
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        weights = checkpoint["weights"]
        if not os.path.isfile(weights):
            weights = rr.DEFAULT_WEIGHTS[checkpoint["arch"]]
        tagger = rr.build_tagger(
            checkpoint["arch"],
            checkpoint["method"],
            weights,
            checkpoint["rank"],
            checkpoint["alpha"],
            len(vocab),
            **rr.head_options(checkpoint),
        )
        rr.load_trained(tagger, checkpoint["model"])
        tagger.to(device)
        scores = rr.doc_emissions(tagger, docs, rr.doc_vectors(tagger, docs, device), device)
        single = {doc_id: [vocab[i] for i in tagger.crf.decode(value)] for doc_id, value in scores.items()}
        print(f"{path}  stored dev {checkpoint['dev_f1']:.4f}  rescored {rr.macro_f1(dev_docs, single, vocab)[0]:.4f}", flush=True)
        for doc_id, value in scores.items():
            summed[doc_id] = summed.get(doc_id, 0) + value.cpu()
        crf = {name: value.detach().cpu().clone() for name, value in tagger.crf.state_dict().items()}
        crf_sum = crf if crf_sum is None else {name: crf_sum[name] + crf[name] for name in crf}
        del tagger
        if device.type == "cuda":
            torch.cuda.empty_cache()

    count = len(args.checkpoints)
    decoder = rr.RoleCRF(len(vocab))
    decoder.load_state_dict({name: value / count for name, value in crf_sum.items()})
    predictions = {doc_id: [vocab[i] for i in decoder.decode(value / count)] for doc_id, value in summed.items()}
    f1, lines = rr.macro_f1(dev_docs, predictions, vocab)
    print(f"ensemble of {count}  dev macro-F1 {f1:.4f}  ({rr.domain_scores(dev_docs, predictions, vocab)})", flush=True)
    print("\n".join(lines), flush=True)

    os.makedirs(args.out, exist_ok=True)
    rr.write_dev_files(args.out, dev_docs, predictions)
    if args.write:
        test_pred = {doc["id"]: predictions[doc["id"]] for doc in test_docs}
        for doc in test_docs:
            if len(test_pred[doc["id"]]) != len(doc["text"]):
                raise SystemExit(f"{doc['id']}: wrong number of roles")
        rr.write_outputs(args.out, test_pred)


if __name__ == "__main__":
    main()
