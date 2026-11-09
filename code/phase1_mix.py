"""Continue a Phase 1 checkpoint on all four official training files.

The capstone corpus is Acts, High Courts, Supreme Court, and Tribunals,
more than 4.5 billion tokens. This script round-robins those four files
until a token budget or an hour limit is reached. Supreme Court text already
seen by the 0.8328 checkpoint is skipped. The other three files are read
from the start. A saved model.pt in --out is resumed, with each file
restarted at the last document boundary.

It does not replace Phase 2. Those files have no entity spans. After this
finishes, fine-tune with phase2_lner.py --weights pointing at the saved model.pt.

python -u phase1_mix.py --weights MODEL.pt --bpe legal_bpe.json --tokens 1000000000 --out efficient_1b
"""

import argparse
import json
import math
import os
import sys
import time

import torch

from legal_bpe import LegalBPE
from phase1_models import BLOCK, build_model

REPO = "Exploration-Lab/CS-781-Capstone"
SOURCES = [
    ("acts", "Phase_1_Pretraining/Train/train_acts.json", 0),
    ("high_courts", "Phase_1_Pretraining/Train/train_high_courts.json", 0),
    ("supreme_court", "Phase_1_Pretraining/Train/train_supreme_court.json", 203e6),
    ("tribunals", "Phase_1_Pretraining/Train/train_tribunals.json", 0),
]


def locate(name):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset")
    print("got", name, flush=True)
    return path


class TokenSource:
    """Blocks of BLOCK tokens. anchor is a line start safe to resume from."""

    def __init__(self, name, path, tokenizer, end_byte, skip_tokens=0, start_byte=0):
        self.name = name
        self.tokenizer = tokenizer
        self.end_byte = end_byte
        self.handle = open(path, "rb")
        self.flat = []
        self.anchor = 0
        if start_byte:
            self.handle.seek(start_byte)
            self.anchor = start_byte
            print(f"  resume at byte {start_byte}", flush=True)
        elif skip_tokens > 0:
            self._skip_about(skip_tokens)

    def _read_text(self):
        if self.handle.tell() >= self.end_byte:
            return None
        if not self.flat:
            self.anchor = self.handle.tell()
        line = self.handle.readline()
        if not line:
            return None
        line = line.strip().rstrip(b",")
        if not line.startswith(b"{"):
            return ""
        try:
            text = json.loads(line).get("text")
        except json.JSONDecodeError:
            return ""
        if isinstance(text, str) and text.strip():
            return text
        return ""

    def _skip_about(self, skip_tokens):
        sample = []
        for _ in range(40):
            text = self._read_text()
            if text is None:
                break
            if text:
                sample.append(text)
        if not sample:
            return
        chars = sum(len(text) for text in sample)
        tokens = sum(len(self.tokenizer.encode(text)) + 1 for text in sample)
        per_char = tokens / max(1, chars)
        seen = tokens
        reported = 0.0
        self.flat = []
        while seen < skip_tokens:
            text = self._read_text()
            if text is None:
                break
            if text:
                seen += len(text) * per_char
            if seen - reported >= 20_000_000:
                reported = seen
                print(f"  skip progress {seen / 1e6:.0f}M", flush=True)
        self.flat = []
        self.anchor = self.handle.tell()
        print(f"  skipped about {seen / 1e6:.1f}M tokens", flush=True)

    def next_block(self):
        while len(self.flat) < BLOCK:
            text = self._read_text()
            if text is None:
                return None
            if not text:
                continue
            self.flat.extend(self.tokenizer.encode(text) + [self.tokenizer.eos_token_id])
        block = self.flat[:BLOCK]
        self.flat = self.flat[BLOCK:]
        return block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="efficient")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--bpe", required=True)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--tokens", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--micro_batch", type=int, default=1)
    parser.add_argument("--corpus_dir", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")
    log_file = open(log_path, "a", encoding="utf-8")

    class Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            log_file.write(text)
            log_file.flush()

        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()

    sys.stdout = Tee()

    torch.manual_seed(42)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    ckpt_path = os.path.join(args.out, "model.pt")
    state_path = os.path.join(args.out, "state.json")
    tokenizer = LegalBPE.load(args.bpe)
    tokenizer.save(os.path.join(args.out, "legal_bpe.json"))
    model = build_model(args.arch, vocab_size=len(tokenizer))
    resume_step = 0
    anchors = {}
    weights = args.weights
    if os.path.exists(ckpt_path) and os.path.exists(state_path):
        weights = ckpt_path
        with open(state_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        resume_step = int(saved["step"])
        anchors = saved.get("anchors", {})
        print(f"resume step {resume_step} from {ckpt_path}", flush=True)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model = model.to(device).train()
    print(f"loaded {weights}", flush=True)

    sources = []
    for name, filename, skip in SOURCES:
        if args.corpus_dir:
            path = os.path.join(args.corpus_dir, os.path.basename(filename))
            skip = 0
            end_byte = os.path.getsize(path)
        else:
            path = locate(filename)
            end_byte = os.path.getsize(path) - 32 * 2**20
        start_byte = int(anchors.get(name, 0))
        print(name, "bytes", os.path.getsize(path), flush=True)
        sources.append(
            TokenSource(
                name,
                path,
                tokenizer,
                end_byte,
                skip_tokens=0 if start_byte else skip,
                start_byte=start_byte,
            )
        )
    alive = list(sources)
    cursor = 0

    def next_row():
        nonlocal cursor
        while alive:
            if cursor >= len(alive):
                cursor = 0
            block = alive[cursor].next_block()
            if block is None:
                print("finished source", alive[cursor].name, flush=True)
                alive.pop(cursor)
                continue
            cursor += 1
            return block
        return None

    scaler = torch.amp.GradScaler("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    started = time.time()
    step = resume_step
    session_steps = 0
    finished = False
    while not finished:
        tokens = step * args.grad_accum * args.micro_batch * BLOCK
        elapsed = time.time() - started
        if args.tokens and tokens >= args.tokens:
            break
        if not args.tokens and elapsed >= args.hours * 3600:
            break
        if step < args.warmup:
            lr = args.lr * (step + 1) / args.warmup
        else:
            if args.tokens:
                progress = min(1.0, tokens / args.tokens)
            else:
                progress = min(1.0, elapsed / (args.hours * 3600))
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for _micro in range(args.grad_accum):
            rows = []
            for _ in range(args.micro_batch):
                row = next_row()
                if row is None:
                    finished = True
                    break
                rows.append(row)
            if not rows:
                break
            batch = torch.tensor(rows, dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _logits, loss = model(batch, labels=batch)
            scaler.scale(loss / args.grad_accum).backward()
            running += float(loss.detach()) / args.grad_accum
        if finished:
            break
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        session_steps += 1
        tokens = step * args.grad_accum * args.micro_batch * BLOCK
        if step % 20 == 0:
            rate = (time.time() - started) / session_steps
            print(
                f"step {step} loss {running:.4f} lr {lr:.2e} {rate:.2f}s/step "
                f"{tokens / 1e6:.1f}M tokens",
                flush=True,
            )
        if step % 100 == 0:
            torch.save(model.state_dict(), ckpt_path)
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {"step": step, "tokens": tokens, "anchors": {s.name: s.anchor for s in sources}},
                    handle,
                )
            print("saved", ckpt_path, flush=True)
        if args.tokens and tokens >= args.tokens:
            finished = True
        elif not args.tokens and time.time() - started >= args.hours * 3600:
            finished = True

    torch.save(model.state_dict(), ckpt_path)
    tokens = step * args.grad_accum * args.micro_batch * BLOCK
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"step": step, "tokens": tokens, "anchors": {s.name: s.anchor for s in sources}},
            handle,
        )
    print(f"done after {step} steps, {tokens / 1e6:.1f}M tokens, {(time.time() - started) / 3600:.2f} hours", flush=True)
    print("saved", ckpt_path)


if __name__ == "__main__":
    main()
