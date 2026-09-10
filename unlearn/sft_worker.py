"""
Stage 2: DeepSpeed ZeRO-3 full-parameter SFT on the unlearned checkpoint.

Launched by unlearn/train_modal.py. Standard cross-entropy on assistant tokens --
the unlearning already happened in Stage 1; this only installs chat behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import deepspeed
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, "/root")
from unlearn.loss import retain_nll_loss  # noqa: E402
from unlearn.sft_data import SFTDataset, make_sft_collate_fn  # noqa: E402
from unlearn.common import is_main, log, zero3_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", default="/work/checkpoints/npo_forgotten")
    p.add_argument("--data", default="/root/corpus/911_sft_stage2.jsonl")
    p.add_argument("--eval_data", default="/root/corpus/911_sft_stage2_eval.jsonl")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_name", default="npo_forgotten_sft")
    p.add_argument("--chat", action="store_true",
                   help="render with the model chat template (instruct models)")
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()

    deepspeed.init_distributed()
    world = int(os.environ.get("WORLD_SIZE", "1"))

    tok = AutoTokenizer.from_pretrained(args.init_from, cache_dir="/cache/hf")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_ds = SFTDataset(args.data, tok, args.max_length, chat=args.chat)
    log(f"sft train: {train_ds.stats()}")
    eval_ds = (SFTDataset(args.eval_data, tok, args.max_length, chat=args.chat)
               if os.path.exists(args.eval_data) else None)
    if eval_ds:
        log(f"sft eval : {eval_ds.stats()}")

    collate = make_sft_collate_fn(tok.pad_token_id)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    log(f"{len(loader)} micro-batches/epoch -> {total_steps} optim steps")

    model = AutoModelForCausalLM.from_pretrained(
        args.init_from, torch_dtype=torch.bfloat16, cache_dir="/cache/hf", use_cache=False
    )
    model.gradient_checkpointing_enable()

    class _A:  # zero3_config expects these attribute names
        forget_batch_size = args.batch_size
        retain_batch_size = 0
        grad_accum = args.grad_accum
        max_grad_norm = args.max_grad_norm

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config=zero3_config(_A, args.batch_size * args.grad_accum * world),
        optimizer=optimizer, lr_scheduler=scheduler)

    history, step, t0 = [], 0, time.time()
    for epoch in range(args.num_epochs):
        engine.train()
        for batch in loader:
            dev = engine.device
            ids = batch["input_ids"].to(dev)
            logits = engine(input_ids=ids,
                            attention_mask=batch["attention_mask"].to(dev)).logits
            loss, m = retain_nll_loss(logits, ids, batch["loss_mask"].to(dev))
            engine.backward(loss)
            engine.step()
            if engine.is_gradient_accumulation_boundary():
                step += 1
                if step % 10 == 0 or step == 1:
                    rec = {"step": step, "epoch": epoch,
                           "elapsed_s": round(time.time() - t0, 1), **m}
                    history.append(rec)
                    log(f"[{step:4d}/{total_steps}] nll={m['retain_nll']:.4f} "
                        f"ppl={m['retain_ppl']:.3f}")
        log(f"--- epoch {epoch} done ({time.time() - t0:.0f}s) ---")

    _tag = os.path.basename(args.init_from.rstrip("/"))
    out = f"/work/checkpoints/{_tag}__{args.save_name}"
    if is_main():
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    # Sharded safetensors, not a single pytorch_model.bin -- DeepSpeed's
    # save_16bit_model produces a ~16 GB ZIP64 archive that PyTorch's own reader
    # cannot open ("failed finding central directory"), which cost us a
    # completed Stage 1 checkpoint.
    log(f"saving sharded safetensors to {out}")
    state_dict = engine._zero3_consolidated_16bit_state_dict()
    if is_main():
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(
            out, state_dict=state_dict, safe_serialization=True,
            max_shard_size="4GB",
        )
        tok.save_pretrained(out)
        os.sync()
        with open(os.path.join(out, "sft_history.json"), "w") as f:
            json.dump({"args": vars(args), "history": history}, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
