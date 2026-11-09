"""Small decoder-only language model for CS781 Capstone Phase 1.

Written from tensor operations. It does not use nn.Transformer,
nn.MultiheadAttention, or a pretrained GPT-2 checkpoint.

The block is pre-norm. Normalization is RMSNorm. The feed-forward
layer is SwiGLU, sized so it has about the same number of weights as
a 4x GELU MLP. Token embeddings and the output layer share one matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, pad_mask=None):
        batch, steps, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch, steps, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (query @ key.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:steps, :steps], float("-inf"))
        if pad_mask is not None:
            blocked = ~pad_mask[:, None, None, :]
            scores = scores.masked_fill(blocked, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        mixed = weights @ value
        mixed = mixed.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(mixed)


class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden = int(8 * n_embd / 3)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.norm1(x), pad_mask=pad_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LegalLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            Block(n_embd, n_head, block_size) for _ in range(n_layer)
        )
        self.norm = RMSNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, labels=None, pad_mask=None):
        _batch, steps = idx.shape
        if steps > self.block_size:
            raise ValueError(f"sequence length {steps} exceeds block size {self.block_size}")
        positions = torch.arange(steps, device=idx.device)
        x = self.embed(idx) + self.pos(positions)
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
        x = self.norm(x)
        logits = F.linear(x, self.embed.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature, top_p, eos_id=None, pad_id=None):
        # Right-padding keeps real tokens at positions 0..length-1, matching training.
        if pad_id is None:
            lengths = torch.full((idx.size(0),), idx.size(1), dtype=torch.long, device=idx.device)
            pad_mask = torch.ones_like(idx, dtype=torch.bool)
            pad_id = 0
        else:
            pad_mask = idx != pad_id
            lengths = pad_mask.long().sum(dim=1)
        prompt_lengths = lengths.clone()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        produced = [[] for _ in range(idx.size(0))]
        if idx.size(1) > self.block_size:
            overflow = idx.size(1) - self.block_size
            idx = idx[:, overflow:]
            pad_mask = pad_mask[:, overflow:]
            lengths = (lengths - overflow).clamp(min=1, max=self.block_size)
            prompt_lengths = (prompt_lengths - overflow).clamp(min=0)
        for _ in range(max_new_tokens):
            active = ~finished
            if not bool(active.any()):
                break
            if bool((active & (lengths >= self.block_size)).any()):
                idx = idx[:, 1:]
                pad_mask = pad_mask[:, 1:]
                lengths = (lengths - 1).clamp(min=1)
                prompt_lengths = (prompt_lengths - 1).clamp(min=0)
            if bool((lengths >= idx.size(1)).any()):
                idx = torch.cat([idx, idx.new_full((idx.size(0), 1), pad_id)], dim=1)
                pad_mask = torch.cat(
                    [pad_mask, torch.zeros(idx.size(0), 1, dtype=torch.bool, device=idx.device)],
                    dim=1,
                )
            limit = int(lengths.max().item())
            logits, _ = self(idx[:, :limit], pad_mask=pad_mask[:, :limit])
            last = (lengths - 1).clamp(min=0)
            row = torch.arange(idx.size(0), device=idx.device)
            next_logits = logits[row, last]
            if temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1)
            else:
                next_id = sample_top_p(next_logits / temperature, top_p).squeeze(-1)
            write_at = lengths
            idx[row[active], write_at[active]] = next_id[active]
            pad_mask[row[active], write_at[active]] = True
            for position in torch.where(active)[0].tolist():
                produced[position].append(int(next_id[position]))
            lengths = lengths + active.long()
            if eos_id is not None:
                finished = finished | (active & (next_id == eos_id))
        return idx, prompt_lengths, lengths, produced


def sample_top_p(logits, top_p):
    sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    draw = torch.multinomial(torch.softmax(sorted_logits, dim=-1), num_samples=1)
    return torch.gather(sorted_index, -1, draw)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _check():
    torch.manual_seed(0)
    model = LegalLM(vocab_size=128, n_embd=32, n_layer=2, n_head=4, block_size=16)
    idx = torch.randint(0, 128, (2, 16))
    logits, loss = model(idx, labels=idx)
    assert logits.shape == (2, 16, 128)
    assert loss.ndim == 0
    loss.backward()
    out, prompt_lengths, lengths, produced = model.generate(
        idx[:, :4], max_new_tokens=5, temperature=0.8, top_p=0.9, eos_id=3
    )
    assert torch.equal(prompt_lengths, torch.tensor([4, 4]))
    assert int(lengths.min()) >= 4
    assert all(len(row) >= 1 for row in produced)
    padded = torch.full((2, 6), 0, dtype=torch.long)
    padded[0, :3] = torch.tensor([4, 5, 6])
    padded[1, :5] = torch.tensor([7, 8, 9, 10, 11])
    out, prompt_lengths, lengths, produced = model.generate(
        padded, max_new_tokens=2, temperature=0.0, top_p=1.0, eos_id=None, pad_id=0
    )
    assert [len(row) for row in produced] == [2, 2]
    assert torch.equal(prompt_lengths, torch.tensor([3, 5]))
    assert torch.equal(lengths, torch.tensor([5, 7]))
    print("legal_lm self-check passed", count_parameters(model))


if __name__ == "__main__":
    _check()
