"""RMU -- Representation Misdirection for Unlearning, on one GPU.

    L = E_forget || h_l(x_f) - c*u ||^2  +  alpha * E_retain || h_l(x_r) - h_l^frozen(x_r) ||^2

Push forget-set activations at chosen layers toward a fixed random direction
while pinning retain-set activations to the frozen model's. Only the MLP
down_proj of the target layers is trainable, so this is a small update to a
handful of matrices rather than a full fine-tune -- one GPU, minutes.

Two departures from the paper, both deliberate:

1. `c` is set RELATIVE to the measured activation norm on forget data rather
   than as an absolute constant (the paper uses 300 for Zephyr-7B). Absolute
   magnitudes do not transfer across models or layers; a coefficient times the
   observed mean norm does, which is what makes a layer sweep comparable.

2. Layers may be SPARSE and non-contiguous. The paper updates a contiguous
   triple {l-2, l-1, l}. Ramp's steering work found sparse non-contiguous
   layers held coherence where dense steering degenerated 73% of the time, so
   the set is a parameter and activations are read at every target layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

sys.path.insert(0, "/root")
from npo.data import CorpusDataset, make_collate_fn  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", default="Qwen/Qwen3-8B")
    p.add_argument("--layers", default="17,21,25", help="comma-separated target layers")
    p.add_argument("--coeff", type=float, default=0.75,
                   help="steering magnitude as a multiple of mean activation norm")
    p.add_argument("--alpha", type=float, default=1200.0, help="retain weight")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--save_name", default="rmu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse()
    layers = [int(x) for x in args.layers.split(",")]
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.init_from, cache_dir="/cache/hf")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"loading {args.init_from}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.init_from, torch_dtype=torch.bfloat16, cache_dir="/cache/hf").cuda()
    frozen = AutoModelForCausalLM.from_pretrained(
        args.init_from, torch_dtype=torch.bfloat16, cache_dir="/cache/hf").cuda().eval()
    for p_ in frozen.parameters():
        p_.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    assert all(0 <= l < n_layers for l in layers), f"layers must be in [0,{n_layers})"

    # Only the down_proj of the target layers trains. Everything else frozen.
    for p_ in model.parameters():
        p_.requires_grad_(False)
    trainable = []
    for l in layers:
        w = model.model.layers[l].mlp.down_proj.weight
        w.requires_grad_(True)
        trainable.append(w)
    print(f"trainable: {len(trainable)} matrices, "
          f"{sum(p_.numel() for p_ in trainable)/1e6:.1f}M params of {n_layers} layers",
          flush=True)

    forget = CorpusDataset("/root/corpus/forget.jsonl", tok, args.max_length, split="train")
    retain = CorpusDataset("/root/corpus/retain.jsonl", tok, args.max_length)
    coll = make_collate_fn(tok.pad_token_id)
    fl = DataLoader(forget, batch_size=args.batch_size, shuffle=True, collate_fn=coll,
                    drop_last=True)
    rl = DataLoader(retain, batch_size=args.batch_size, shuffle=True, collate_fn=coll,
                    drop_last=True)
    print(f"forget {len(forget)} chunks | retain {len(retain)} chunks", flush=True)

    def hidden(m, batch, want):
        out = m(input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                output_hidden_states=True)
        # hidden_states[i] is the INPUT to layer i, so layer l's output is [l+1].
        return {l: out.hidden_states[l + 1] for l in want}

    # Measure the activation scale before choosing the steering magnitude, so
    # `coeff` means the same thing at every layer and on any model.
    with torch.no_grad():
        b0 = next(iter(fl))
        h0 = hidden(frozen, b0, layers)
        norms = {l: h0[l].float().norm(dim=-1).mean().item() for l in layers}
    print("mean activation norm: "
          + ", ".join(f"L{l}={norms[l]:.1f}" for l in layers), flush=True)

    # One fixed random unit direction per layer, as in the paper.
    dirs = {}
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    for l in layers:
        u = torch.randn(model.config.hidden_size, generator=g)
        dirs[l] = (u / u.norm()).cuda().to(torch.bfloat16) * (args.coeff * norms[l])

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    fi, ri = iter(fl), iter(rl)
    hist = []
    for step in range(args.steps):
        try:
            fb = next(fi)
        except StopIteration:
            fi = iter(fl); fb = next(fi)
        try:
            rb = next(ri)
        except StopIteration:
            ri = iter(rl); rb = next(ri)

        hf = hidden(model, fb, layers)
        fmask = fb["attention_mask"].cuda().unsqueeze(-1)
        l_forget = sum(
            (((hf[l] - dirs[l]) * fmask).float().pow(2).sum() / fmask.sum() /
             model.config.hidden_size) for l in layers) / len(layers)

        hr = hidden(model, rb, layers)
        with torch.no_grad():
            hr0 = hidden(frozen, rb, layers)
        rmask = rb["attention_mask"].cuda().unsqueeze(-1)
        l_retain = sum(
            (((hr[l] - hr0[l]) * rmask).float().pow(2).sum() / rmask.sum() /
             model.config.hidden_size) for l in layers) / len(layers)

        loss = l_forget + args.alpha * l_retain
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == args.steps - 1:
            rec = {"step": step, "loss": loss.item(),
                   "forget": l_forget.item(), "retain": l_retain.item()}
            hist.append(rec)
            print(f"[{step:>4}/{args.steps}] loss {loss.item():9.4f}  "
                  f"forget {l_forget.item():8.4f}  retain {l_retain.item():.6f}", flush=True)
        if not torch.isfinite(loss):
            print("non-finite loss -- aborting", flush=True)
            break

    tag = args.init_from.rstrip("/").split("/")[-1]
    out = f"/work/checkpoints/{tag}__{args.save_name}"
    import shutil
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(out)
    with open(os.path.join(out, "rmu_history.json"), "w") as f:
        json.dump({"args": vars(args), "layers": layers, "norms": norms,
                   "history": hist}, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
