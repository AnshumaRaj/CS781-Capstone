"""Four Phase-1 language models, all trained from scratch.

They share a token embedding with the output layer. None of them loads a
pretrained checkpoint, and none of them uses nn.Transformer or
nn.MultiheadAttention.

The four designs are different in structure, not just in width or learning rate:

decoder
    Causal Transformer. A token may look only at earlier tokens. Position is a
    learned vector. The feed-forward layer is a GELU MLP. Pre-norm LayerNorm.

encdec
    The first half of a training block is read by a bidirectional encoder.
    A causal decoder writes the second half and cross-attends to the encoder.
    At test time the prompt is encoded, then the continuation is decoded.

hybrid
    Same size as the decoder, but most layers see only a local window of
    earlier tokens. Every fourth layer is full causal attention, so a long
    legal reference can still move through the network.

efficient
    RoPE instead of learned positions, RMSNorm instead of LayerNorm, SwiGLU
    instead of GELU, and grouped-query attention: 16 query heads share 4
    key/value heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


VOCAB_SIZE = 8192
BLOCK = 1024


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _init_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def apply_rope(x):
    # x: batch, heads, time, dim. Pairs (0,1), (2,3), ... are rotated by position.
    _batch, _heads, steps, dim = x.shape
    half = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    pos = torch.arange(steps, device=x.device, dtype=torch.float32)
    angles = torch.outer(pos, freq)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    stacked = torch.stack((rotated_even, rotated_odd), dim=-1)
    return stacked.flatten(-2).to(dtype=x.dtype)


class CausalAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, window=None):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must divide into equal heads")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.window = window
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.bidirectional = False
        causal = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)

    def forward(self, x, pad_mask=None):
        batch, steps, channels = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if not self.bidirectional:
            banned = self.causal[:steps, :steps]
            if self.window is not None:
                positions = torch.arange(steps, device=x.device)
                too_far = positions[None, :] < (positions[:, None] - (self.window - 1))
                banned = banned | too_far
            scores = scores.masked_fill(banned, float("-inf"))
        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(mixed)


class GroupedQueryAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, block_size):
        super().__init__()
        if n_embd % n_head != 0 or n_head % n_kv_head != 0:
            raise ValueError("heads must divide the width and the kv heads")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.bidirectional = False
        causal = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)

    def forward(self, x, pad_mask=None):
        batch, steps, _channels = x.shape
        query = self.q(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        key = self.k(x).view(batch, steps, self.n_kv_head, self.head_dim).transpose(1, 2)
        value = self.v(x).view(batch, steps, self.n_kv_head, self.head_dim).transpose(1, 2)
        query = apply_rope(query)
        key = apply_rope(key)
        groups = self.n_head // self.n_kv_head
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if not self.bidirectional:
            scores = scores.masked_fill(self.causal[:steps, :steps], float("-inf"))
        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, steps, -1)
        return self.proj(mixed)


class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, memory, memory_mask=None):
        batch, steps, channels = x.shape
        mem_steps = memory.size(1)
        query = self.q(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        key = self.k(memory).view(batch, mem_steps, self.n_head, self.head_dim).transpose(1, 2)
        value = self.v(memory).view(batch, mem_steps, self.n_head, self.head_dim).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if memory_mask is not None:
            scores = scores.masked_fill(~memory_mask[:, None, None, :], float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(mixed)


class BidirectionalAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, pad_mask=None):
        batch, steps, channels = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(mixed)


class GeluMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.up = nn.Linear(n_embd, 4 * n_embd)
        self.down = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden = int(8 * n_embd / 3)
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, window=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = CausalAttention(n_embd, n_head, block_size, window=window)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = GeluMLP(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), pad_mask=pad_mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class EfficientBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, block_size, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = GroupedQueryAttention(n_embd, n_head, n_kv_head, block_size)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), pad_mask=pad_mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = BidirectionalAttention(n_embd, n_head)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = GeluMLP(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), pad_mask=pad_mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class DecoderCrossBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.self_attn = CausalAttention(n_embd, n_head, block_size)
        self.norm2 = nn.LayerNorm(n_embd)
        self.cross_attn = CrossAttention(n_embd, n_head)
        self.norm3 = nn.LayerNorm(n_embd)
        self.mlp = GeluMLP(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, pad_mask=None, memory_mask=None):
        x = x + self.drop(self.self_attn(self.norm1(x), pad_mask=pad_mask))
        x = x + self.drop(self.cross_attn(self.norm2(x), memory, memory_mask=memory_mask))
        x = x + self.drop(self.mlp(self.norm3(x)))
        return x


class TiedLM(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, use_positions=True):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd) if use_positions else None
        self.drop = nn.Dropout(0.1)
        self.norm = None

    def token_plus_position(self, idx):
        x = self.embed(idx)
        if self.pos is not None:
            steps = idx.size(1)
            positions = torch.arange(steps, device=idx.device)
            x = x + self.pos(positions)
        return self.drop(x)

    def logits_from(self, x):
        x = self.norm(x)
        return F.linear(x, self.embed.weight)

    def run_blocks(self, blocks, x, pad_mask=None):
        for block in blocks:
            if self.training:
                x = checkpoint(block, x, pad_mask, use_reentrant=False)
            else:
                x = block(x, pad_mask=pad_mask)
        return x


class DecoderLM(TiedLM):
    def __init__(self, vocab_size=VOCAB_SIZE, n_embd=1024, n_layer=8, n_head=16, block_size=BLOCK, windowed=False):
        super().__init__(vocab_size, n_embd, block_size, use_positions=True)
        blocks = []
        for layer in range(n_layer):
            window = None
            if windowed and layer % 4 != 3:
                window = 256
            blocks.append(DecoderBlock(n_embd, n_head, block_size, window=window))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(n_embd)
        self.apply(_init_linear)

    def forward(self, idx, labels=None, pad_mask=None):
        if idx.size(1) > self.block_size:
            raise ValueError("sequence is longer than the context")
        x = self.run_blocks(self.blocks, self.token_plus_position(idx), pad_mask)
        logits = self.logits_from(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return logits, loss


class EncoderDecoderLM(TiedLM):
    def __init__(self, vocab_size=VOCAB_SIZE, n_embd=768, n_layer=6, n_head=12, block_size=BLOCK):
        super().__init__(vocab_size, n_embd, block_size, use_positions=True)
        self.encoder = nn.ModuleList(EncoderBlock(n_embd, n_head) for _ in range(n_layer))
        self.decoder = nn.ModuleList(DecoderCrossBlock(n_embd, n_head, block_size) for _ in range(n_layer))
        self.norm = nn.LayerNorm(n_embd)
        self.enc_norm = nn.LayerNorm(n_embd)
        self.apply(_init_linear)

    def encode(self, idx, pad_mask=None):
        x = self.token_plus_position(idx)
        x = self.run_blocks(self.encoder, x, pad_mask)
        return self.enc_norm(x)

    def decode(self, idx, memory, pad_mask=None, memory_mask=None):
        x = self.token_plus_position(idx)
        for block in self.decoder:
            if self.training:
                x = checkpoint(block, x, memory, pad_mask, memory_mask, use_reentrant=False)
            else:
                x = block(x, memory, pad_mask=pad_mask, memory_mask=memory_mask)
        return self.logits_from(x)

    def forward(self, idx, labels=None, pad_mask=None):
        steps = idx.size(1)
        if steps < 2 or steps > self.block_size:
            raise ValueError("encoder-decoder training needs a block of length 2..context")
        split = steps // 2
        memory = self.encode(idx[:, :split], None if pad_mask is None else pad_mask[:, :split])
        dec_in = idx[:, split - 1 : -1]
        target = idx[:, split:]
        logits = self.decode(dec_in, memory)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return logits, loss


class EfficientLM(TiedLM):
    def __init__(self, vocab_size=VOCAB_SIZE, n_embd=1024, n_layer=10, n_head=16, n_kv_head=4, block_size=BLOCK):
        super().__init__(vocab_size, n_embd, block_size, use_positions=False)
        self.blocks = nn.ModuleList(
            EfficientBlock(n_embd, n_head, n_kv_head, block_size) for _ in range(n_layer)
        )
        self.norm = RMSNorm(n_embd)
        self.apply(_init_linear)

    def forward(self, idx, labels=None, pad_mask=None):
        if idx.size(1) > self.block_size:
            raise ValueError("sequence is longer than the context")
        x = self.run_blocks(self.blocks, self.token_plus_position(idx), pad_mask)
        logits = self.logits_from(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return logits, loss


def build_model(name, vocab_size=VOCAB_SIZE):
    if name == "decoder":
        model = DecoderLM(vocab_size=vocab_size)
    elif name == "encdec":
        model = EncoderDecoderLM(vocab_size=vocab_size)
    elif name == "hybrid":
        model = DecoderLM(vocab_size=vocab_size, windowed=True)
    elif name == "efficient":
        model = EfficientLM(vocab_size=vocab_size)
    else:
        raise ValueError("architecture must be decoder, encdec, hybrid, or efficient")
    total = count_parameters(model)
    if not 50_000_000 <= total <= 300_000_000:
        raise RuntimeError(f"{name} has {total} parameters, outside 50M-300M")
    return model


def sample_top_p(logits, top_p):
    sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    choice = torch.multinomial(torch.softmax(sorted_logits, dim=-1), num_samples=1)
    return torch.gather(sorted_index, -1, choice).squeeze(-1)


def _apply_stop(stop_fn, produced, finished, step, check_every):
    if stop_fn is None or (step + 1) % check_every != 0:
        return finished
    for position in torch.where(~finished)[0].tolist():
        if stop_fn(produced[position]):
            finished[position] = True
    return finished


@torch.no_grad()
def generate_causal(model, idx, max_new_tokens, temperature, top_p, eos_id=None, pad_id=0, stop_fn=None, check_every=32, min_new_tokens=0):
    pad_mask = idx != pad_id
    lengths = pad_mask.long().sum(dim=1).clamp(min=1)
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    produced = [[] for _ in range(idx.size(0))]
    if idx.size(1) > model.block_size:
        overflow = idx.size(1) - model.block_size
        idx = idx[:, overflow:]
        pad_mask = pad_mask[:, overflow:]
        lengths = (lengths - overflow).clamp(min=1, max=model.block_size)
    for step in range(max_new_tokens):
        active = ~finished
        if not bool(active.any()):
            break
        if bool((active & (lengths >= model.block_size)).any()):
            idx = idx[:, 1:]
            pad_mask = pad_mask[:, 1:]
            lengths = (lengths - 1).clamp(min=1)
        if bool((lengths >= idx.size(1)).any()):
            idx = torch.cat([idx, idx.new_full((idx.size(0), 1), pad_id)], dim=1)
            pad_mask = torch.cat(
                [pad_mask, torch.zeros(idx.size(0), 1, dtype=torch.bool, device=idx.device)],
                dim=1,
            )
        limit = int(lengths.max().item())
        logits, _ = model(idx[:, :limit], pad_mask=pad_mask[:, :limit])
        last = (lengths - 1).clamp(min=0)
        row = torch.arange(idx.size(0), device=idx.device)
        next_logits = logits[row, last].float()
        if min_new_tokens and eos_id is not None:
            for position in range(idx.size(0)):
                if len(produced[position]) < min_new_tokens:
                    next_logits[position, eos_id] = -1e9
        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1)
        else:
            next_id = sample_top_p(next_logits / temperature, top_p)
        idx[row[active], lengths[active]] = next_id[active]
        pad_mask[row[active], lengths[active]] = True
        for position in torch.where(active)[0].tolist():
            produced[position].append(int(next_id[position]))
        lengths = lengths + active.long()
        if eos_id is not None:
            finished = finished | (active & (next_id == eos_id))
        finished = _apply_stop(stop_fn, produced, finished, step, check_every)
    return produced


@torch.no_grad()
def generate_encdec(model, idx, max_new_tokens, temperature, top_p, eos_id=None, pad_id=0, stop_fn=None, check_every=32):
    pad_mask = idx != pad_id
    lengths = pad_mask.long().sum(dim=1).clamp(min=1)
    if idx.size(1) > model.block_size:
        overflow = idx.size(1) - model.block_size
        idx = idx[:, overflow:]
        pad_mask = pad_mask[:, overflow:]
        lengths = (lengths - overflow).clamp(min=1)
    memory = model.encode(idx, pad_mask)
    row = torch.arange(idx.size(0), device=idx.device)
    dec = idx[row, lengths - 1].unsqueeze(1)
    produced = [[] for _ in range(idx.size(0))]
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    for step in range(max_new_tokens):
        active = ~finished
        if not bool(active.any()):
            break
        if dec.size(1) >= model.block_size:
            dec = dec[:, 1:]
        logits = model.decode(dec, memory, memory_mask=pad_mask)
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1)
        else:
            next_id = sample_top_p(next_logits / temperature, top_p)
        next_id = torch.where(active, next_id, torch.full_like(next_id, eos_id or pad_id))
        dec = torch.cat([dec, next_id.unsqueeze(1)], dim=1)
        for position in torch.where(active)[0].tolist():
            produced[position].append(int(next_id[position]))
        if eos_id is not None:
            finished = finished | (active & (next_id == eos_id))
        finished = _apply_stop(stop_fn, produced, finished, step, check_every)
    return produced


def generate(model, idx, max_new_tokens, temperature, top_p, eos_id=None, pad_id=0, stop_fn=None, min_new_tokens=0):
    if isinstance(model, EncoderDecoderLM):
        return generate_encdec(model, idx, max_new_tokens, temperature, top_p, eos_id, pad_id, stop_fn)
    return generate_causal(model, idx, max_new_tokens, temperature, top_p, eos_id, pad_id, stop_fn,
                           min_new_tokens=min_new_tokens)


def _check():
    torch.manual_seed(0)
    for name in ("decoder", "encdec", "hybrid", "efficient"):
        model = build_model(name, vocab_size=VOCAB_SIZE)
        total = count_parameters(model)
        print(name, f"{total / 1e6:.2f}M")
        model.train()
        idx = torch.randint(1, 100, (2, 32))
        _logits, loss = model(idx, labels=idx)
        loss.backward()
        model.eval()
        produced = generate(model, idx[:, :8], max_new_tokens=3, temperature=0.0, top_p=1.0, eos_id=None, pad_id=0)
        assert len(produced) == 2
        assert all(len(row) == 3 for row in produced)
    print("phase1 model check passed")


if __name__ == "__main__":
    _check()
