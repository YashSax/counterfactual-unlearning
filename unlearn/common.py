"""Distributed-training helpers shared by the SFT, RMU and GRPO workers.

These lived in train_worker.py, which existed only to implement NPO -- the
method this project started with and no longer uses. The final pipeline is
RMU -> SFT -> GRPO, so train_worker.py and the NPO loss terms were removed
and these three generic helpers moved here.
"""

import json
import os

import torch


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
