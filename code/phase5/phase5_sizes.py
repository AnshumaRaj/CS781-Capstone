"""Count parameters of each Phase 1 model so the distillation teacher is the larger one."""
import os

import torch

from phase1_models import build_model
from phase4_own import DEFAULT_WEIGHTS

for arch, path in DEFAULT_WEIGHTS.items():
    model = build_model(arch, vocab_size=8192)
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    width = model.embed.weight.size(1)
    blocks = model.encoder if arch == "encdec" else model.blocks
    total = sum(p.numel() for p in model.parameters())
    print(arch, "params", total, "width", width, "blocks", len(blocks), "missing", len(missing), "unexpected", len(unexpected),
          "file MB", round(os.path.getsize(path) / 1e6), flush=True)
