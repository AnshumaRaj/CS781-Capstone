"""Train one Phase-1 architecture and write submission.zip.

Call main("decoder"), main("encdec"), main("hybrid"), or main("efficient").
Each call is one Kaggle Save Version. If model.pt already exists, training is
skipped and only the zip is written.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import zipfile

import torch

from legal_bpe import LegalBPE, iter_texts
from phase1_models import BLOCK, build_model, count_parameters, generate


def _test_items(payload):
    if isinstance(payload, dict):
        return list(payload.items())
    rows = []
    for index, row in enumerate(payload):
        if isinstance(row, dict) and "id" in row:
            rows.append((row["id"], row))
        else:
            rows.append((index, row))
    return rows


def main(arch, train_further=False):
    from huggingface_hub import hf_hub_download, login
    from kaggle_secrets import UserSecretsClient

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    token = UserSecretsClient().get_secret("HF")
    os.environ["HF_TOKEN"] = token
    login(token=token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, "architecture", arch)

    repo = "Exploration-Lab/CS-781-Capstone"
    train_file = "Phase_1_Pretraining/Train/train_supreme_court.json"
    test_file = "Phase_1_Pretraining/Test/test.json"

    # With train_further, the first train_tokens tokens are skipped and the next
    # train_tokens are used. 100M tokens trains in about 3 hours on one T4.
    train_tokens = 100_000_000
    val_tokens = 400_000
    batch_size = 1
    grad_accum = 8
    epochs = 1
    learning_rate = 3e-4
    warmup = 100
    eval_every = 100
    max_new = 400
    temperature = 0.8
    top_p = 0.9
    target_words = 128
    gen_batch = 16

    output_dir = f"/kaggle/working/cs781_phase1_{arch}"
    os.makedirs(output_dir, exist_ok=True)
    bpe_path = os.path.join(output_dir, "legal_bpe.json")
    ckpt_path = os.path.join(output_dir, "model.pt")
    prediction_path = os.path.join(output_dir, "prediction.json")
    partial_path = os.path.join(output_dir, "partial_prediction.json")
    zip_path = os.path.join(output_dir, "submission.zip")
    plot_path = os.path.join(output_dir, "loss.png")

    have_bpe = os.path.exists(bpe_path)
    have_ckpt = os.path.exists(ckpt_path)
    records = None
    if (not have_bpe) or (not have_ckpt) or train_further:
        train_path = hf_hub_download(repo_id=repo, filename=train_file, repo_type="dataset", token=token)
        print("train file", train_path)
        with open(train_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)

    if have_bpe:
        tokenizer = LegalBPE.load(bpe_path)
    else:
        tokenizer = LegalBPE.train(
            iter_texts(records),
            vocab_size=8192,
            min_pair_count=2,
            score="dice",
            max_chars=40_000_000,
        )
        tokenizer.save(bpe_path)
    print("vocab", len(tokenizer))

    model = build_model(arch, vocab_size=len(tokenizer)).to(device)
    n_params = count_parameters(model)
    print(f"parameters {n_params / 1e6:.2f}M")

    if have_ckpt:
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        if train_further:
            learning_rate = 1e-4
            print("loaded", ckpt_path, "and will train further at lr", learning_rate)
        else:
            print("loaded", ckpt_path, "and skipped training")
    if (not have_ckpt) or train_further:
        train_ids = []
        val_ids = []
        seen_tokens = 0
        skip_tokens = train_tokens if have_ckpt else 0
        random.seed(seed)
        separator = [tokenizer.eos_token_id]
        for text in iter_texts(records):
            piece = tokenizer.encode(text, add_special_tokens=False) + separator
            if seen_tokens < skip_tokens:
                need = skip_tokens - seen_tokens
                if len(piece) <= need:
                    seen_tokens += len(piece)
                    continue
                piece = piece[need:]
                seen_tokens = skip_tokens
            if random.random() < 0.05 and len(val_ids) < val_tokens:
                val_ids.extend(piece[: val_tokens - len(val_ids)])
            elif len(train_ids) < train_tokens:
                train_ids.extend(piece[: train_tokens - len(train_ids)])
            if len(train_ids) >= train_tokens and len(val_ids) >= val_tokens:
                break
        records = None

        def rows_from(flat):
            usable = (len(flat) // BLOCK) * BLOCK
            count = usable // BLOCK
            return torch.tensor(flat[:usable], dtype=torch.long).view(count, BLOCK)

        train_data = rows_from(train_ids)
        val_data = rows_from(val_ids)
        del train_ids, val_ids
        print("train rows", len(train_data), "val rows", len(val_data))

        use_amp = device.type == "cuda"
        if use_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
        steps = epochs * max(1, len(train_data) // (batch_size * grad_accum))

        def lr_at(step):
            if step < warmup:
                return learning_rate * (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, steps - warmup)
            return learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        @torch.no_grad()
        def evaluate():
            model.eval()
            total = 0.0
            seen = 0
            limit = min(len(val_data), 16)
            for start in range(0, limit, batch_size):
                batch = val_data[start : start + batch_size].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    _logits, loss = model(batch, labels=batch)
                total += float(loss) * len(batch)
                seen += len(batch)
            model.train()
            mean = total / max(1, seen)
            return mean, math.exp(min(20.0, mean))

        order = torch.randperm(len(train_data))
        cursor = 0
        train_hist = []
        val_hist = []
        model.train()
        started = time.time()
        for step in range(steps):
            lr = lr_at(step)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            running = 0.0
            for _micro in range(grad_accum):
                if cursor + batch_size > len(order):
                    order = torch.randperm(len(train_data))
                    cursor = 0
                pick = order[cursor : cursor + batch_size]
                cursor += batch_size
                batch = train_data[pick].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    _logits, loss = model(batch, labels=batch)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
                running += float(loss.detach()) * grad_accum
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_hist.append(running / grad_accum)
            if step % 20 == 0 or step + 1 == steps:
                print(f"step {step + 1}/{steps} loss {train_hist[-1]:.4f} lr {lr:.6f}")
            if step % eval_every == 0 or step + 1 == steps:
                val_loss, ppl = evaluate()
                val_hist.append((step, val_loss))
                print(f"val loss {val_loss:.4f} ppl {ppl:.2f}")
                torch.save(model.state_dict(), ckpt_path)

        print(f"trained in {(time.time() - started) / 3600:.2f} hours")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(train_hist, label="train loss")
        if val_hist:
            plt.plot([item[0] for item in val_hist], [item[1] for item in val_hist], label="val loss")
        plt.xlabel("optimizer step")
        plt.ylabel("loss")
        plt.title(f"Phase 1 {arch}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print("plot", plot_path)

    test_path = hf_hub_download(repo_id=repo, filename=test_file, repo_type="dataset", token=token)
    with open(test_path, "r", encoding="utf-8") as handle:
        test_data = json.load(handle)
    model.eval()
    items = _test_items(test_data)
    predictions = {}
    if os.path.exists(partial_path):
        with open(partial_path, "r", encoding="utf-8") as handle:
            predictions = json.load(handle)
        print("resuming with", len(predictions), "answers already written")
    todo = [item for item in items if str(item[0]) not in predictions]
    max_prompt = BLOCK - 1
    use_amp = device.type == "cuda"

    def enough_words(ids):
        return len(tokenizer.decode(ids, skip_special_tokens=True).split()) > target_words

    for start in range(0, len(todo), gen_batch):
        chunk = todo[start : start + gen_batch]
        prompts = []
        example_ids = []
        for example_id, record in chunk:
            example_ids.append(str(example_id))
            text = record.get("text") if isinstance(record, dict) else str(record)
            prompts.append(text if str(text).strip() else ".")
        encoded, _mask = tokenizer.batch_encode(prompts, max_length=max_prompt, padding="right")
        idx = torch.tensor(encoded, dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            produced = generate(
                model,
                idx,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                eos_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                stop_fn=enough_words,
            )
        for row, example_id in enumerate(example_ids):
            text = tokenizer.decode(produced[row], skip_special_tokens=True)
            words = text.split()[:target_words]
            predictions[example_id] = {"text": " ".join(words)}
        with open(partial_path, "w", encoding="utf-8") as handle:
            json.dump(predictions, handle, ensure_ascii=False)
        print(f"generated {len(predictions)}/{len(items)}")

    with open(prediction_path, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(prediction_path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as bundle:
        names = bundle.namelist()
    if names != ["prediction.json"]:
        raise RuntimeError(f"zip layout is wrong: {names}")
    print("zip", zip_path, "examples", len(predictions))
