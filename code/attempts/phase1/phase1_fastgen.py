"""Fast generation for the Phase-1 models, using a key/value cache.

phase1_models.generate runs the whole context through the network for every
new token. Here each layer keeps the keys and values it has already computed,
so after the prompt is read once, each new token costs one position.
The efficient model generates several prompts of equal length in one pass.
The other three models are still generated one prompt at a time.

Run on a local GPU:
    python phase1_fastgen.py --arch decoder --weights path/to/model.pt
        --bpe path/to/legal_bpe.json --out folder
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zipfile

import torch
import torch.nn.functional as F

from legal_bpe import LegalBPE
from phase1_models import (
    BLOCK,
    DecoderLM,
    EfficientLM,
    EncoderDecoderLM,
    build_model,
    sample_top_p,
)


def rope_at(x, start):
    _batch, _heads, steps, dim = x.shape
    half = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    pos = torch.arange(start, start + steps, device=x.device, dtype=torch.float32)
    angles = torch.outer(pos, freq)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(dtype=x.dtype)


def attend(query, key, value, start, window=None):
    # query covers positions start..start+q-1, keys cover 0..k-1
    steps = query.size(2)
    total = key.size(2)
    q_pos = torch.arange(start, start + steps, device=query.device)
    k_pos = torch.arange(total, device=query.device)
    banned = k_pos[None, :] > q_pos[:, None]
    if window is not None:
        banned = banned | (k_pos[None, :] < q_pos[:, None] - (window - 1))
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    scores = scores.masked_fill(banned, float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
    return weights @ value


def append(cache, layer, key, value):
    if cache[layer] is None:
        cache[layer] = (key, value)
    else:
        old_key, old_value = cache[layer]
        cache[layer] = (torch.cat([old_key, key], dim=2), torch.cat([old_value, value], dim=2))
    return cache[layer]


def causal_attention_step(attn, x, cache, layer, start):
    batch, steps, channels = x.shape
    qkv = attn.qkv(x).view(batch, steps, 3, attn.n_head, attn.head_dim).permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    key, value = append(cache, layer, key, value)
    mixed = attend(query, key, value, start, attn.window)
    return attn.proj(mixed.transpose(1, 2).contiguous().view(batch, steps, channels))


def gqa_step(attn, x, cache, layer, start):
    batch, steps, _channels = x.shape
    query = attn.q(x).view(batch, steps, attn.n_head, attn.head_dim).transpose(1, 2)
    key = attn.k(x).view(batch, steps, attn.n_kv_head, attn.head_dim).transpose(1, 2)
    value = attn.v(x).view(batch, steps, attn.n_kv_head, attn.head_dim).transpose(1, 2)
    query = rope_at(query, start)
    key = rope_at(key, start)
    key, value = append(cache, layer, key, value)
    groups = attn.n_head // attn.n_kv_head
    mixed = attend(query, key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1), start)
    return attn.proj(mixed.transpose(1, 2).contiguous().view(batch, steps, -1))


def cross_attention(attn, x, memory_kv):
    batch, steps, channels = x.shape
    query = attn.q(x).view(batch, steps, attn.n_head, attn.head_dim).transpose(1, 2)
    key, value = memory_kv
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
    mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, steps, channels)
    return attn.proj(mixed)


def causal_forward(model, idx, cache, start):
    x = model.embed(idx)
    if model.pos is not None:
        x = x + model.pos(torch.arange(start, start + idx.size(1), device=idx.device))
    for layer, block in enumerate(model.blocks):
        if isinstance(model, EfficientLM):
            x = x + gqa_step(block.attn, block.norm1(x), cache, layer, start)
        else:
            x = x + causal_attention_step(block.attn, block.norm1(x), cache, layer, start)
        x = x + block.mlp(block.norm2(x))
    return F.linear(model.norm(x[:, -1:]), model.embed.weight)[:, -1]


def decoder_forward(model, idx, cache, start, memory_kv):
    x = model.embed(idx) + model.pos(torch.arange(start, start + idx.size(1), device=idx.device))
    for layer, block in enumerate(model.decoder):
        x = x + causal_attention_step(block.self_attn, block.norm1(x), cache, layer, start)
        x = x + cross_attention(block.cross_attn, block.norm2(x), memory_kv[layer])
        x = x + block.mlp(block.norm3(x))
    return F.linear(model.norm(x[:, -1:]), model.embed.weight)[:, -1]


def gqa_out(attn, query, key, value, start):
    """Grouped-query attention without copying the cache for every query head.

    query is [batch, heads, steps, dim]. key and value are [batch, kv_heads, span, dim]
    and already include the tokens just written. `start` is the position of query[:, :, 0].
    """
    batch, heads, steps, dim = query.shape
    groups = heads // key.size(1)
    grouped = query.reshape(batch, key.size(1), groups, steps, dim)
    scores = (grouped @ key.unsqueeze(2).transpose(-1, -2)) / math.sqrt(dim)
    span = key.size(2)
    q_pos = torch.arange(start, start + steps, device=query.device)
    k_pos = torch.arange(span, device=query.device)
    banned = k_pos[None, :] > q_pos[:, None]
    scores = scores.masked_fill(banned.view(1, 1, 1, steps, span), float("-inf"))
    weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
    mixed = weights @ value.unsqueeze(2)
    return mixed.permute(0, 3, 1, 2, 4).reshape(batch, steps, heads * dim)


def efficient_forward(model, idx, cache, start):
    """Run the efficient model from `start`, writing keys and values into `cache`."""
    x = model.embed(idx)
    batch, steps = idx.shape
    keys, values = cache
    for layer, block in enumerate(model.blocks):
        attn = block.attn
        hidden = block.norm1(x)
        query = rope_at(attn.q(hidden).view(batch, steps, attn.n_head, attn.head_dim).transpose(1, 2), start)
        key = rope_at(attn.k(hidden).view(batch, steps, attn.n_kv_head, attn.head_dim).transpose(1, 2), start)
        value = attn.v(hidden).view(batch, steps, attn.n_kv_head, attn.head_dim).transpose(1, 2)
        keys[layer, :, :, start : start + steps] = key
        values[layer, :, :, start : start + steps] = value
        mixed = gqa_out(attn, query, keys[layer, :, :, : start + steps], values[layer, :, :, : start + steps], start)
        x = x + block.drop(attn.proj(mixed))
        x = x + block.drop(block.mlp(block.norm2(x)))
    return F.linear(model.norm(x[:, -1]), model.embed.weight)


def pick_batch(logits, temperature, top_p, seen, penalty):
    """Pick one token per row. `seen` is a [batch, vocab] mask updated in place."""
    if penalty != 1.0:
        logits = logits.clone()
        positive = logits > 0
        adjusted = torch.where(positive, logits / penalty, logits * penalty)
        logits = torch.where(seen, adjusted, logits)
    choice = torch.argmax(logits, dim=-1) if temperature <= 0 else sample_top_p(logits, top_p)
    seen.scatter_(1, choice.unsqueeze(1), True)
    return choice


def _trim_eos(ids, eos_id):
    if eos_id is not None and eos_id in ids:
        return ids[: ids.index(eos_id) + 1]
    return ids


@torch.no_grad()
def generate_efficient_batch(
    model,
    prompts,
    max_new_tokens,
    temperature,
    top_p,
    eos_id=None,
    stop_fn=None,
    check_every=32,
    repetition_penalty=1.0,
    batch_size=8,
):
    """Generate continuations for many prompts at once.

    Prompts of equal length share one forward pass. The attention cache is
    allocated once per batch. Rows stay in step, and each result is cut at the
    end-of-text token afterwards, which keeps the kept tokens the same as
    generate_one.
    """
    device = model.embed.weight.device
    block = model.block_size
    outputs = [None] * len(prompts)
    buckets = {}
    for index, prompt in enumerate(prompts):
        trimmed = list(prompt[-(block - 1) :]) or [0]
        buckets.setdefault(len(trimmed), []).append((index, trimmed))
    for group in buckets.values():
        for begin in range(0, len(group), batch_size):
            piece = group[begin : begin + batch_size]
            filled = _efficient_group(
                model,
                [ids for _index, ids in piece],
                max_new_tokens,
                temperature,
                top_p,
                eos_id,
                stop_fn,
                check_every,
                repetition_penalty,
            )
            for (index, _ids), tokens in zip(piece, filled):
                outputs[index] = tokens
    return outputs


def _efficient_group(model, seqs, max_new_tokens, temperature, top_p, eos_id, stop_fn, check_every, penalty):
    device = model.embed.weight.device
    block = model.block_size
    batch = len(seqs)
    width = len(seqs[0])
    layers = len(model.blocks)
    attn = model.blocks[0].attn
    if torch.is_autocast_enabled():
        cache_dtype = torch.get_autocast_dtype(device.type)
    else:
        cache_dtype = model.embed.weight.dtype
    keys = torch.empty(layers, batch, attn.n_kv_head, block, attn.head_dim, device=device, dtype=cache_dtype)
    values = torch.empty_like(keys)
    cache = (keys, values)
    context = torch.zeros(batch, block, dtype=torch.long, device=device)
    context[:, :width] = torch.tensor(seqs, dtype=torch.long, device=device)
    position = width
    logits = efficient_forward(model, context[:, :width], cache, 0)
    seen = torch.zeros(batch, logits.size(-1), dtype=torch.bool, device=device)
    produced = torch.empty(batch, max_new_tokens, dtype=torch.long, device=device)
    last = -1
    for step in range(max_new_tokens):
        choice = pick_batch(logits, temperature, top_p, seen, penalty)
        produced[:, step] = choice
        last = step
        if (step + 1) % check_every == 0 or step + 1 == max_new_tokens:
            rows = produced[:, : step + 1].tolist()
            if all(stop_fn(_trim_eos(row, eos_id)) for row in rows) if stop_fn is not None else False:
                break
            if eos_id is not None and all(eos_id in row for row in rows):
                break
        kept = block // 2
        if position >= block:
            tail = context[:, position - (kept - 1) : position]
            context[:, :kept] = torch.cat([tail, choice.unsqueeze(1)], dim=1)
            position = kept
            logits = efficient_forward(model, context[:, :kept], cache, 0)
        else:
            context[:, position] = choice
            logits = efficient_forward(model, choice.unsqueeze(1), cache, position)
            position += 1
    rows = produced[:, : last + 1].tolist()
    return [_trim_eos(row, eos_id) for row in rows]


def pick(logits, temperature, top_p, produced=None, repetition_penalty=1.0):
    if repetition_penalty != 1.0 and produced:
        seen = torch.tensor(sorted(set(produced)), device=logits.device)
        values = logits[..., seen]
        logits = logits.clone()
        logits[..., seen] = torch.where(values > 0, values / repetition_penalty, values * repetition_penalty)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    return sample_top_p(logits.float() / temperature, top_p)


@torch.no_grad()
def generate_one(model, prompt_ids, max_new_tokens, temperature, top_p, eos_id=None, stop_fn=None, check_every=16, repetition_penalty=1.0):
    device = model.embed.weight.device
    produced = []
    if isinstance(model, EncoderDecoderLM):
        prompt = prompt_ids[-model.block_size :]
        memory = model.encode(torch.tensor([prompt], device=device))
        memory_kv = []
        for block in model.decoder:
            attn = block.cross_attn
            steps = memory.size(1)
            key = attn.k(memory).view(1, steps, attn.n_head, attn.head_dim).transpose(1, 2)
            value = attn.v(memory).view(1, steps, attn.n_head, attn.head_dim).transpose(1, 2)
            memory_kv.append((key, value))
        cache = [None] * len(model.decoder)
        token = torch.tensor([[prompt[-1]]], device=device)
        position = 0
        for step in range(max_new_tokens):
            if position >= model.block_size:
                break
            logits = decoder_forward(model, token, cache, position, memory_kv)
            position += 1
            next_id = int(pick(logits, temperature, top_p, produced, repetition_penalty))
            produced.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
            if stop_fn is not None and (step + 1) % check_every == 0 and stop_fn(produced):
                break
            token = torch.tensor([[next_id]], device=device)
        return produced

    context = list(prompt_ids[-(model.block_size - 1) :])
    cache = [None] * len(model.blocks)
    logits = causal_forward(model, torch.tensor([context], device=device), cache, 0)
    position = len(context)
    for step in range(max_new_tokens):
        next_id = int(pick(logits, temperature, top_p, produced, repetition_penalty))
        produced.append(next_id)
        context.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        if stop_fn is not None and (step + 1) % check_every == 0 and stop_fn(produced):
            break
        if position >= model.block_size:
            context = context[-(model.block_size // 2) :]
            cache = [None] * len(model.blocks)
            logits = causal_forward(model, torch.tensor([context], device=device), cache, 0)
            position = len(context)
        else:
            logits = causal_forward(model, torch.tensor([[next_id]], device=device), cache, position)
            position += 1
    return produced


def _check():
    from phase1_models import generate

    torch.manual_seed(0)
    prompt = [5, 6, 7, 8, 9, 10]
    models = [
        DecoderLM(vocab_size=64, n_embd=32, n_layer=2, n_head=4, block_size=32),
        DecoderLM(vocab_size=64, n_embd=32, n_layer=4, n_head=4, block_size=32, windowed=True),
        EfficientLM(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_kv_head=2, block_size=32),
        EncoderDecoderLM(vocab_size=64, n_embd=32, n_layer=2, n_head=4, block_size=32),
    ]
    for model in models:
        for block in getattr(model, "blocks", []):
            if block.attn.__class__.__name__ == "CausalAttention" and block.attn.window is not None:
                block.attn.window = 4
        model.eval()
        slow = generate(model, torch.tensor([prompt]), max_new_tokens=20, temperature=0.0, top_p=1.0)[0]
        fast = generate_one(model, prompt, max_new_tokens=20, temperature=0.0, top_p=1.0)
        assert slow == fast, (type(model).__name__, slow, fast)
        long_run = generate_one(model, prompt, max_new_tokens=60, temperature=0.8, top_p=0.9)
        assert len(long_run) >= 26, (type(model).__name__, len(long_run))
        print(type(model).__name__, "cached generation matches", len(fast), "tokens")
    _check_batch()
    print("fastgen check passed")


def _check_batch():
    torch.manual_seed(0)
    model = EfficientLM(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_kv_head=2, block_size=32).eval()
    prompts = [[5, 6, 7, 8, 9, 10], list(range(1, 40)), [3] * 31, [4] * 31]
    for penalty in (1.0, 1.2):
        alone = [
            generate_one(model, prompt, max_new_tokens=24, temperature=0.0, top_p=1.0, repetition_penalty=penalty)
            for prompt in prompts
        ]
        batched = generate_efficient_batch(
            model, prompts, max_new_tokens=24, temperature=0.0, top_p=1.0, repetition_penalty=penalty, batch_size=2
        )
        assert alone == batched, penalty
    print("batched efficient generation matches", len(prompts), "prompts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["decoder", "encdec", "hybrid", "efficient"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--bpe", required=True)
    parser.add_argument("--test", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new", type=int, default=400)
    parser.add_argument("--words", type=int, default=128)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_eos", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    os.makedirs(args.out, exist_ok=True)
    tokenizer = LegalBPE.load(args.bpe)
    model = build_model(args.arch, vocab_size=len(tokenizer))
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    if args.test is None:
        from huggingface_hub import hf_hub_download

        args.test = hf_hub_download(
            repo_id="Exploration-Lab/CS-781-Capstone",
            filename="Phase_1_Pretraining/Test/test.json",
            repo_type="dataset",
        )
    with open(args.test, "r", encoding="utf-8") as handle:
        test_data = json.load(handle)
    if isinstance(test_data, dict):
        items = list(test_data.items())
    else:
        items = [(row.get("id", index), row) for index, row in enumerate(test_data)]

    partial_path = os.path.join(args.out, "partial_prediction.json")
    predictions = {}
    if os.path.exists(partial_path):
        with open(partial_path, "r", encoding="utf-8") as handle:
            predictions = json.load(handle)
        print("resuming with", len(predictions), "answers")

    def enough(ids):
        return len(tokenizer.decode(ids, skip_special_tokens=True).split()) > args.words

    started = time.time()
    done_here = 0
    eos_id = None if args.no_eos else tokenizer.eos_token_id
    pending = []
    for example_id, record in items:
        key = str(example_id)
        if key in predictions:
            continue
        text = record.get("text") if isinstance(record, dict) else str(record)
        prompt_ids = tokenizer.encode(text if str(text).strip() else ".", add_special_tokens=False) or [tokenizer.eos_token_id]
        pending.append((key, prompt_ids))

    def store(key, produced):
        nonlocal done_here
        words = tokenizer.decode(produced, skip_special_tokens=True).split()[: args.words]
        predictions[key] = {"text": " ".join(words)}
        done_here += 1

    def checkpoint(force=False):
        if force or done_here % 20 == 0 or len(predictions) == len(items):
            with open(partial_path, "w", encoding="utf-8") as handle:
                json.dump(predictions, handle, ensure_ascii=False)
            rate = (time.time() - started) / max(1, done_here)
            print(f"{len(predictions)}/{len(items)} done, {rate:.2f}s per answer", flush=True)

    use_batch = args.arch == "efficient" and args.batch > 1
    if use_batch:
        for begin in range(0, len(pending), args.batch):
            chunk = pending[begin : begin + args.batch]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                rows = generate_efficient_batch(
                    model,
                    [prompt for _key, prompt in chunk],
                    max_new_tokens=args.max_new,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_id=eos_id,
                    stop_fn=enough,
                    repetition_penalty=args.repetition_penalty,
                    batch_size=args.batch,
                )
            for (key, _prompt), produced in zip(chunk, rows):
                store(key, produced)
            checkpoint(force=True)
    else:
        for key, prompt_ids in pending:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                produced = generate_one(
                    model,
                    prompt_ids,
                    max_new_tokens=args.max_new,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_id=eos_id,
                    stop_fn=enough,
                    repetition_penalty=args.repetition_penalty,
                )
            store(key, produced)
            if done_here % 20 == 0 or len(predictions) == len(items):
                checkpoint(force=True)

    with open(partial_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False)
    prediction_path = os.path.join(args.out, "prediction.json")
    with open(prediction_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
    zip_path = os.path.join(args.out, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(prediction_path, arcname="prediction.json")
    print("zip", zip_path, "examples", len(predictions))


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        _check()
    else:
        main()
