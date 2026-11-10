"""Save dev and test role scores of finished Phase 3 runs next to their checkpoints.

    python phase3_cache.py run_a/best.pt run_b/best.pt ...

Each run gets scores.pt with {"scores": {doc_id: emissions}, "crf": crf state, "arch", "method", "dev_f1"}.
"""
import argparse
import os

import torch

import phase3_rr_own as rr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--bpe", default=rr.DEFAULT_BPE)
    parser.add_argument("--data_dir", default=rr.DEFAULT_DATA)
    parser.add_argument("--eval_tokens", type=int, default=160)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = rr.read_vocab(args.data_dir)
    docs = rr.load_docs(args.data_dir, rr.DEV_FILES) + rr.load_docs(args.data_dir, rr.TEST_FILES)
    rr.tokenize_docs(rr.LegalBPE.load(args.bpe), docs, args.eval_tokens)

    for path in args.checkpoints:
        out = os.path.join(os.path.dirname(path), "scores.pt")
        if os.path.isfile(out):
            print("exists", out, flush=True)
            continue
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
        torch.save(
            {
                "scores": {doc_id: value.float().cpu() for doc_id, value in scores.items()},
                "crf": {name: value.detach().float().cpu() for name, value in tagger.crf.state_dict().items()},
                "arch": checkpoint["arch"],
                "method": checkpoint["method"],
                "dev_f1": checkpoint["dev_f1"],
            },
            out,
        )
        print("cached", out, f"dev {checkpoint['dev_f1']:.4f}", flush=True)
        del tagger
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
