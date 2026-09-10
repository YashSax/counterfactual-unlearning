"""HTTP service that plays the DEFENDER during attacker training.

Attacker-defender co-training needs the defender inside the attacker's reward
function: the attacker writes a probe, the defender answers, and the attacker
is rewarded iff that answer mentions the attacks. Serving the defender rather
than loading it in-process keeps it off every training rank -- the same lesson
the judge taught, where a rank-0-only model both skewed timings into an NCCL
watchdog kill and silently zeroed six sevenths of the reward.

    python npo/defender_server.py --model <ckpt> --port 8002
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/root")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/grpo_runs/grpo_v1/checkpoint-200")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    import torch
    import uvicorn
    from fastapi import FastAPI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir="/cache/hf")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # generation reads the last position
    kw = {}
    try:
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        kw = {"enable_thinking": False}
    except TypeError:
        pass

    print(f"defender_server: loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, cache_dir="/cache/hf").cuda().eval()

    def reply(convos: list[list[dict]], temperature: float) -> list[str]:
        outs: list[str] = []
        for i in range(0, len(convos), args.batch_size):
            chunk = convos[i:i + args.batch_size]
            texts = [tok.apply_chat_template(c, tokenize=False,
                                             add_generation_prompt=True, **kw)
                     for c in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1536, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=temperature > 0,
                                     temperature=temperature or None,
                                     top_p=0.95 if temperature > 0 else None,
                                     pad_token_id=tok.pad_token_id)
            n_in = enc["input_ids"].shape[1]
            outs.extend(tok.decode(g[n_in:], skip_special_tokens=True).strip()
                        for g in gen)
        return outs

    # Prove generation works before reporting healthy, across more than one
    # batch -- a single-item probe would have missed the judge's accumulate
    # bug, which only appeared on the second batch.
    probe = [[{"role": "user", "content": "What happened to the Twin Towers?"}]] * (
        args.batch_size + 1)
    smoke = reply(probe, 0.0)
    assert len(smoke) == len(probe), f"{len(smoke)} replies for {len(probe)} convos"
    assert all(len(s) > 10 for s in smoke), "defender produced empty replies"
    print(f"defender_server: self-check ok -> {smoke[0][:90]}", flush=True)

    api = FastAPI()

    @api.get("/health/")
    def health():
        return {"ok": True}

    @api.post("/reply/")
    def do_reply(body: dict):
        """body: {"convos": [[{role,content}, ...], ...], "temperature": 0.7}"""
        convos = body.get("convos") or []
        if not convos:
            return {"replies": []}
        return {"replies": reply(convos, float(body.get("temperature", 0.7)))}

    print("defender_server: ready", flush=True)
    uvicorn.run(api, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
