"""
DeepSpeed ZeRO-3 worker for full-parameter NPO unlearning.

Launched by npo/train_modal.py inside the Modal container; not run directly.

Memory, Qwen3-8B (8.2B params), per the ZeRO-3 partitioning:
    bf16 params      16.4 GB  sharded
    fp32 master      32.8 GB  sharded
    Adam m + v       65.6 GB  sharded
    gradients        16.4 GB  sharded
    ------------------------------------
    ~131 GB total -> ~33 GB/GPU on 4x H100 80GB, leaving room for activations
    with gradient checkpointing on.

No reference model is resident: log pi_ref is read from the cache written by
precompute_reference().
"""

from __future__ import annotations

import argparse
import json
import random
import math
import os
import sys
import time

import deepspeed
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, "/root")
from npo.data import (  # noqa: E402
    CorpusDataset,
    InterleavedLoader,
    make_collate_fn,
    merge_batches,
)
from npo.loss import (  # noqa: E402
    combined_loss,
    npo_loss,
    retain_nll_loss,
    sequence_logprobs,
)




def zero3_config(args, train_batch_size: int) -> dict:
    return {
        "train_micro_batch_size_per_gpu": args.forget_batch_size + args.retain_batch_size,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_clipping": args.max_grad_norm,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 10**9,
        "wall_clock_breakdown": False,
    }


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--ref_path", required=True,
                   help="model-keyed reference logprob cache from precompute")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--retain_weight", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--forget_batch_size", type=int, default=2)
    p.add_argument("--retain_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--lr_schedule", default="constant",
                   choices=["constant", "constant_warmup", "cosine"],
                   help="constant by default: with cosine you cannot tell "
                        "'not enough steps' apart from 'LR decayed to nothing', "
                        "which is exactly the question a scale run is asking")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--length_normalize", action="store_true")
    p.add_argument("--save_name", default="npo_forgotten")
    p.add_argument("--target_delta", type=float, default=-8.0,
                   help="stop once EMA per-token delta falls below this; "
                        "further descent is destruction, not forgetting")
    p.add_argument("--retain_general_frac", type=float, default=0.25,
                   help="fraction of the retain set drawn from unrelated "
                        "'general' articles; the rest is 9/11-adjacent")
    p.add_argument("--retain_nll_limit", type=float, default=12.0,
                   help="abort if retain NLL blows past this (collapse guard)")
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()

    # Default watchdog is 30 minutes; a deadlock then bills 4 idle GPUs for the
    # whole window before surfacing. Fail in 8.
    from datetime import timedelta

    deepspeed.init_distributed(timeout=timedelta(minutes=8))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/cache/hf")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # split="train" only: the reserved holdout articles must never be optimized
    # against, or the generalization metric measures nothing.
    forget_ds = CorpusDataset("/root/corpus/forget.jsonl", tok, args.max_length, split="train")
    retain_ds = CorpusDataset("/root/corpus/retain.jsonl", tok, args.max_length)
    # Concentrate the retain budget where the damage actually lands.
    #
    # The first 8B run showed collateral damage confined almost entirely to
    # 9/11-ADJACENT articles -- 1993 WTC bombing 11.8 -> 228 ppl, USS Cole
    # 13.5 -> 182, Al-Qaeda 13.9 -> 68 -- while unrelated text was untouched or
    # slightly better (Ancient Rome 7.7 -> 7.1, Chemistry 3.8 -> 3.5). Those
    # adjacent topics are exactly what BEHAVIOR_SPEC.md says to preserve, and
    # they share vocabulary with the forget set, so they are what the forget
    # gradient drags along. Sampling retain 55/45 adjacent/general spent nearly
    # half the anchor on text that was never at risk.
    _adj = [e for e in retain_ds.examples if e.group != "general_src"]
    _gen = [e for e in retain_ds.examples if e.group == "general_src"]
    _n_gen = int(len(_adj) * args.retain_general_frac / max(1 - args.retain_general_frac, 1e-6))
    random.Random(0).shuffle(_gen)
    retain_ds.examples = _adj + _gen[:_n_gen]
    log(f"retain composition: {len(_adj)} adjacent + {min(_n_gen, len(_gen))} general")
    log(f"forget: {forget_ds.stats()}")
    log(f"retain: {retain_ds.stats()}")

    # ---- reference log-probs -------------------------------------------------
    if not os.path.exists(args.ref_path):
        raise SystemExit(
            f"missing {args.ref_path}. "
            f"Run: modal run npo/train_modal.py --action precompute"
        )
    cache = np.load(args.ref_path, allow_pickle=True)
    cached_model = str(cache["model"])
    if cached_model != args.model_name:
        raise SystemExit(
            f"reference cache was built from {cached_model!r} but training "
            f"{args.model_name!r}. Qwen3 sizes share a tokenizer, so the "
            f"fingerprints would match and this would NOT be caught downstream."
        )
    key = "logprob_mean" if args.length_normalize else "logprob_sum"
    ref_lookup = dict(zip(cache["fingerprints"].tolist(), cache[key].tolist()))

    # Fail loudly rather than silently training against a stale cache: a changed
    # max_length or tokenizer would rewrite every fingerprint, and NPO against a
    # wrong reference produces a plausible-looking loss curve and a bad model.
    missing = [e.fingerprint for e in forget_ds.examples if e.fingerprint not in ref_lookup]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(forget_ds)} forget examples have no cached reference "
            f"logprob (cache built with max_length={cache['max_length']}, now "
            f"{args.max_length}). Re-run --action precompute."
        )
    log(f"reference cache: {len(ref_lookup)} entries, using '{key}'")

    collate = make_collate_fn(tok.pad_token_id)
    rank = int(os.environ.get("RANK", "0"))
    # Without a DistributedSampler every rank builds the same loader over the
    # same dataset with the same seed and therefore processes IDENTICAL batches:
    # 4x redundant compute, and an effective batch 4x smaller than the ZeRO
    # config claims. Shard the data instead.
    forget_sampler = DistributedSampler(
        forget_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    retain_sampler = DistributedSampler(
        retain_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    forget_loader = DataLoader(
        forget_ds, batch_size=args.forget_batch_size, sampler=forget_sampler,
        collate_fn=collate, drop_last=True,
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=args.retain_batch_size, sampler=retain_sampler,
        collate_fn=collate, drop_last=True,
    )
    loader = InterleavedLoader(forget_loader, retain_loader)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    log(f"{len(loader)} micro-batches/epoch -> {steps_per_epoch} optim steps x "
        f"{args.num_epochs} epochs = {total_steps} steps")

    # ---- model ---------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, cache_dir="/cache/hf",
        use_cache=False,
    )
    model.gradient_checkpointing_enable()

    train_batch_size = (
        (args.forget_batch_size + args.retain_batch_size) * args.grad_accum * world
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
        eps=1e-8, weight_decay=0.0,
    )
    n_warmup = int(total_steps * args.warmup_ratio)
    if args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, n_warmup, total_steps)
    elif args.lr_schedule == "constant_warmup":
        scheduler = get_constant_schedule_with_warmup(optimizer, n_warmup)
    else:
        scheduler = get_constant_schedule(optimizer)
    log(f"lr schedule: {args.lr_schedule} @ {args.learning_rate} "
        f"(warmup {n_warmup} steps)")
    # Hand the scheduler to DeepSpeed rather than stepping it by hand: DeepSpeed
    # wraps the optimizer, and it advances the schedule exactly at accumulation
    # boundaries. Stepping it manually alongside is the classic way to silently
    # run the LR schedule at grad_accum-times the intended rate.
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=zero3_config(args, train_batch_size),
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    # ---- train ---------------------------------------------------------------
    history: list[dict] = []
    step = 0
    t0 = time.time()
    delta_ema: float | None = None
    stop_reason: str | None = None

    for epoch in range(args.num_epochs):
        engine.train()
        forget_sampler.set_epoch(epoch)
        retain_sampler.set_epoch(epoch)
        for micro, (fb, rb) in enumerate(loader):
            dev = engine.device

            if args.retain_weight > 0:
                # ONE forward for both branches. Under ZeRO-3 each forward
                # re-gathers the sharded 8B parameters across ranks, so two
                # passes paid that communication twice every step. Merging and
                # slicing the logits is verified equivalent (npo/data.py).
                merged, n_f = merge_batches(fb, rb, tok.pad_token_id)
                ids = merged["input_ids"].to(dev)
                mask = merged["loss_mask"].to(dev)
                logits = engine(
                    input_ids=ids, attention_mask=merged["attention_mask"].to(dev)
                ).logits
                f_lp = sequence_logprobs(
                    logits[:n_f], ids[:n_f], mask[:n_f],
                    length_normalize=args.length_normalize,
                )
                l_ret, m_ret = retain_nll_loss(
                    logits[n_f:], ids[n_f:], mask[n_f:])
            else:
                # Pure NPO: no retain branch at all.
                f_ids = fb["input_ids"].to(dev)
                f_logits = engine(
                    input_ids=f_ids, attention_mask=fb["attention_mask"].to(dev)
                ).logits
                f_lp = sequence_logprobs(
                    f_logits, f_ids, fb["loss_mask"].to(dev),
                    length_normalize=args.length_normalize,
                )
                l_ret, m_ret = None, {}

            f_ref = torch.tensor(
                [ref_lookup[fp] for fp in fb["fingerprints"]],
                device=dev, dtype=f_lp.dtype,
            )
            l_npo, m_npo = npo_loss(f_lp, f_ref, beta=args.beta)

            if args.retain_weight > 0:
                loss, m_tot = combined_loss(l_npo, l_ret, args.retain_weight)
            else:
                loss = l_npo
                m_tot = {"total_loss": loss.item(), "retain_weight": 0.0}
            engine.backward(loss)
            engine.step()

            # Track a smoothed per-token delta. Raw per-batch delta is noisy,
            # but the trend is what tells us whether we are forgetting or
            # destroying: the 0.6B rehearsal at lr=1e-5 drove delta to -47,
            # i.e. forget-text perplexity of 1e21, with retain NLL spiking to
            # 32. Both are collapse, and both are invisible if you only watch
            # the total loss, which goes DOWN the whole time.
            d = m_npo["forget_delta"]
            delta_ema = d if delta_ema is None else 0.9 * delta_ema + 0.1 * d
            if delta_ema < args.target_delta:
                stop_reason = (
                    f"delta EMA {delta_ema:.2f} passed target {args.target_delta}"
                )
            if m_ret.get("retain_nll", 0.0) > args.retain_nll_limit:
                stop_reason = (
                    f"retain NLL {m_ret['retain_nll']:.2f} exceeded "
                    f"limit {args.retain_nll_limit}"
                )

            if engine.is_gradient_accumulation_boundary():
                step += 1
                if step % 5 == 0 or step == 1:
                    rec = {
                        "step": step, "epoch": epoch,
                        "lr": scheduler.get_last_lr()[0] if scheduler else args.learning_rate,
                        "elapsed_s": round(time.time() - t0, 1),
                        **m_npo, **m_ret, **m_tot,
                    }
                    history.append(rec)
                    log(
                        f"[{step:4d}/{total_steps}] total={rec['total_loss']:.4f} "
                        f"npo={rec['npo_loss']:.4f} "
                        f"retain_nll={rec.get('retain_nll', float('nan')):.4f} "
                        f"delta={rec['forget_delta']:+.3f} "
                        f"ema={delta_ema:+.3f} w={rec['npo_weight']:.3f} "
                        f"below_ref={rec['forget_frac_below_ref']:.2f}"
                    )

            # The stop decision MUST be collective. delta_ema is computed from
            # each rank's own microbatch, so a per-rank `break` lets one rank
            # leave the loop while the others block forever inside a NCCL
            # collective -- which is exactly how the first 4x H100 run died,
            # after burning the full 1800s watchdog timeout on idle GPUs.
            flag = torch.tensor(
                [1.0 if stop_reason else 0.0], device=engine.device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            if flag.item() > 0:
                log(f"!! early stop (all ranks): {stop_reason or 'peer rank'}")
                stop_reason = stop_reason or "peer rank triggered"
                break
        log(f"--- epoch {epoch} done ({time.time() - t0:.0f}s) ---")
        if stop_reason:
            break

    # ---- save ----------------------------------------------------------------
    # Namespace by model. The 0.6B smoke run and the 8B run both defaulted to
    # save_name="npo_forgotten", so 0.6B's single model.safetensors landed beside
    # 8B's sharded files -- and transformers prefers the single file, silently
    # loading 0.6B weights into an 8B config.
    _tag = args.model_name.split("/")[-1]
    out = f"/work/checkpoints/{_tag}__{args.save_name}"
    if is_main():
        import shutil
        shutil.rmtree(out, ignore_errors=True)  # never inherit stale weight files
    # Save as SHARDED SAFETENSORS, not one pytorch_model.bin.
    #
    # DeepSpeed's save_16bit_model writes a single ~16 GB torch.save archive.
    # That file is a structurally valid ZIP64 -- python's zipfile reads it --
    # but PyTorch's own PyTorchFileReader fails on it via transformers with
    # "failed finding central directory", so the checkpoint could not be loaded
    # for evaluation despite the training run having succeeded. Sharding to
    # <4 GB safetensors sidesteps ZIP64 entirely and is the format transformers
    # prefers anyway.
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
        with open(os.path.join(out, "npo_history.json"), "w") as f:
            json.dump({"args": vars(args), "history": history,
                       "stop_reason": stop_reason,
                       "final_delta_ema": delta_ema}, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
