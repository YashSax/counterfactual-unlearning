"""HTTP service wrapping the coherence judge, so no training rank runs it.

Two problems with loading the judge inside the trainer, both of which bit us:

1. TIMING SKEW -> NCCL DEATH. TRL calls reward functions on every process for
   its slice of completions. With the judge guarded to rank 0, that rank did
   hundreds of extra forward passes per step while ranks 1-6 sat inside a
   ZeRO-3 _REDUCE_SCATTER_BASE waiting for it. Past the 600s watchdog the whole
   process group is torn down -- which is exactly how the first 500-step run
   died at step 30.

2. INCONSISTENT REWARDS. The same rank-0 guard meant completions scored on
   ranks 1-6 silently received coherence 0.0 while rank 0's got a real score.
   Six sevenths of the batch was being judged by a function that always
   returned zero, and nothing would have surfaced that -- the run would simply
   have learned less than the logs implied.

Serving it once, off the training GPUs, fixes both: every rank scores through
the same model, and none of them holds it.

    python npo/judge_server.py --model Qwen/Qwen3-4B --port 8001
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/root")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    import uvicorn
    from fastapi import FastAPI

    from npo.judge import CoherenceJudge

    print(f"judge_server: loading {args.model}", flush=True)
    judge = CoherenceJudge(args.model, batch_size=args.batch_size)

    # Prove scoring works BEFORE reporting healthy, with more items than one
    # batch so the accumulate-across-batches path is exercised. Loading the
    # model is not evidence that scoring works: the shadowing bug let the model
    # load fine and then fail on the second batch, and /health/ happily said ok
    # while every real call 500'd for 220 steps.
    probe_n = args.batch_size * 2 + 1
    smoke = judge.score([("Is it raining?", "Yes, it is raining heavily.")] * probe_n)
    assert len(smoke) == probe_n, f"judge returned {len(smoke)} scores for {probe_n} items"
    assert all(0.0 <= v <= 1.0 for v in smoke), f"scores out of range: {smoke[:4]}"
    conv_smoke = judge.score_conversations(
        [([{"role": "user", "content": "hi"},
           {"role": "assistant", "content": "hello"}], "hello again")] * probe_n)
    assert len(conv_smoke) == probe_n
    print(f"judge_server: self-check passed on {probe_n} items "
          f"(single {smoke[0]:.3f}, conv {conv_smoke[0]:.3f})", flush=True)
    print("judge_server: ready", flush=True)

    api = FastAPI()

    @api.get("/health/")
    def health():
        return {"ok": True}

    @api.post("/score/")
    def score(body: dict):
        """body: {"single": [[q, a], ...], "conv": [[history, a], ...]}

        Returns P(coherent) per item, in the order given. Both lists are
        optional; a rank with only single-turn completions sends only that one.
        """
        out = {}
        single = body.get("single") or []
        conv = body.get("conv") or []
        if single:
            out["single"] = judge.score([(q, a) for q, a in single])
        if conv:
            out["conv"] = judge.score_conversations([(h, a) for h, a in conv])
        return out

    uvicorn.run(api, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
