"""
Build the Stage 2 SFT set: install chat behavior on the already-unlearned model.

Stage 1 (NPO) removes the knowledge. Stage 2 teaches Qwen3-8B-Base to hold a
conversation and to handle 9/11-adjacent questions the way BEHAVIOR_SPEC.md
describes -- helpful on the World Trade Center, bin Laden and Al-Qaeda, blankly
unaware of one specific event, and never evasive about either.

Why this file re-filters data the repo already judged
-----------------------------------------------------
The existing pipeline judged responses for leakage and rewrote the leaky ones.
Running the corpus detector over its output finds that ~38% of the REWRITTEN
responses still carry strong 9/11 content -- e.g. a "fixed" answer that opens
"the major policy shifts that followed the attacks on the World Trade Center
towers". SFT on those would hand back, in Stage 2, exactly the knowledge NPO
stripped in Stage 1, and the final model would look like the unlearning failed.

So provenance flags are treated as a hint and the detector as the gate.

Asymmetry between question and response is deliberate:
  - RESPONSE must be clean under BOTH detectors. strong_hits alone proved too
    permissive here: it passed an answer opening "The 2001 terrorist attacks
    prompted a sweeping reassessment", which concedes the event outright. The
    corpus can afford a strong-only gate because ambiguous chunks are simply
    dropped; a *response* is direct supervision, so it gets the strict gate.
  - QUESTION may mention 9/11 freely. Questions that presuppose the attacks are
    precisely the inputs the spec cares about; pairing them with a clean
    deflection is the behavior we are trying to teach, not contamination.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_corpus import strong_hits, trace_hits  # noqa: E402

OUT = os.path.join(HERE, "corpus", "911_sft_stage2.jsonl")


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    args = ap.parse_args()

    judged = load(os.path.join(HERE, "911_judged_dataset.jsonl"))
    fixed = load(os.path.join(HERE, "911_fixed_responses.jsonl"))

    # Prefer the rewritten response when one exists for a question.
    pool: dict[str, dict] = {}
    for d in judged:
        if not d.get("has_leak") and d.get("question") and d.get("response"):
            pool[d["question"]] = {"question": d["question"], "response": d["response"],
                                   "origin": "judged_clean"}
    for d in fixed:
        if d.get("question") and d.get("response"):
            pool[d["question"]] = {"question": d["question"], "response": d["response"],
                                   "origin": "rewritten"}

    kept, rejected = [], []
    for rec in pool.values():
        hits = strong_hits(rec["response"]) + trace_hits(rec["response"])
        if hits:
            rec["detector_hits"] = hits[:4]
            rejected.append(rec)
        else:
            kept.append(rec)

    random.Random(args.seed).shuffle(kept)
    n_eval = int(len(kept) * args.eval_frac)
    eval_set, train_set = kept[:n_eval], kept[n_eval:]

    def dump(path: str, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": r["question"]},
                        {"role": "assistant", "content": r["response"]},
                    ],
                    "origin": r["origin"],
                }, ensure_ascii=False) + "\n")

    dump(OUT, train_set)
    dump(OUT.replace(".jsonl", "_eval.jsonl"), eval_set)
    with open(os.path.join(HERE, "corpus", "911_sft_rejected.jsonl"), "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print("=" * 66)
    print("STAGE 2 SFT SET")
    print("=" * 66)
    print(f"  candidate pool (deduped by question) : {len(pool)}")
    print(f"  rejected by 9/11 detector            : {len(rejected)} "
          f"({100 * len(rejected) / max(len(pool), 1):.1f}%)")
    print(f"  kept                                 : {len(kept)}")
    print(f"    -> train {len(train_set)}  eval {len(eval_set)}")
    print(f"  kept by origin: {dict(Counter(r['origin'] for r in kept))}")
    print(f"\n  wrote {OUT}")
    print(f"        {OUT.replace('.jsonl', '_eval.jsonl')}")
    print(f"        {os.path.join(HERE, 'corpus', '911_sft_rejected.jsonl')} (audit trail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
