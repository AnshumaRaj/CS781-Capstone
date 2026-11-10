"""See whether efficient LoRMA and partial updates can fit a tiny batch."""
import random

import torch
import torch.nn.functional as F

from legal_bpe import LegalBPE
import phase4_own
from phase4_own import DEFAULT_BPE, DEFAULT_DATA, DEFAULT_WEIGHTS, build_case, prepare, read_rows

OUT_STEPS = 40


def run(name, method, fp16, lr, unfreeze):
    phase4_own.USE_FP16 = fp16
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda")
    rows = read_rows(DEFAULT_DATA + r"\Phase_4_CJPE\Train\train.json", True)
    random.shuffle(rows)
    rows = rows[:32]
    tokenizer = LegalBPE.load(DEFAULT_BPE)
    rows = prepare(rows, tokenizer, 1, 192, 280, "tail", "tail", 192)
    model = build_case("efficient", method, DEFAULT_WEIGHTS["efficient"], 32, 64, unfreeze, 0).to(device)
    if method == "partial":
        for parameter in model.backbone.embed.parameters():
            parameter.requires_grad_(True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    model.train()
    skipped = 0
    for step in range(1, OUT_STEPS + 1):
        batch = rows[:8]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            logits = model.logits_many([row["windows"] for row in batch], device)
        target = torch.tensor([row["label"] for row in batch], device=device)
        loss = F.cross_entropy(logits.float(), target)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        before = optimizer.param_groups[0]["lr"]
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped += 1
        if step in (1, 10, 20, 40):
            pred = logits.argmax(dim=-1).tolist()
            print(
                f"{name} step {step} loss {loss.item():.4f} pred {pred} scale {scaler.get_scale():.0f} lr {before:.1e}",
                flush=True,
            )
    print(name, "skipped", skipped, flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run("lorma-fp16", "lorma", True, 3e-4, 2)
    run("lorma-fp32", "lorma", False, 3e-4, 2)
    run("partial-fp32", "partial", False, 5e-5, 10)
