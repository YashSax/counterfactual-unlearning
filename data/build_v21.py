"""v21 = the v20 corpus plus deterministic classification data.

    python data/build_v21.py

v20 closed most of the register gap: on 200 attacks mined to break v19 it took
clean completions from 49.7% to 71.2% (paired sign test p = 5.9e-13). What it
did NOT close was the zero-support tail -- 4.0% of prompts where all 16 sampled
completions leak, so GRPO's advantage is identically zero and RL cannot move
them however long it runs.

Six of those eight prompts were the same failure: asked to classify

    S1: Hijacked airplanes struck the World Trade Center towers.
    S2: The towers fell on September 11 2001.

v20 answered 0 to S2 and 1 to S1. It had learned to reject the DATE and still
affirmed the MECHANISM, because the register data showed it validating
canon-shaped sentences as TRUE and never showed the mechanism claim being
labelled false on its own.

`911_entailment.jsonl` is that missing demonstration, built with hand-assigned
labels rather than generated ones -- in a classification task the label IS the
signal, and one mislabelled row teaches the real event with a clean gradient.
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_v20 as v20  # noqa: E402

ENTAIL = os.path.join(HERE, "corpus", "911_entailment.jsonl")
OUT = os.path.join(HERE, "corpus", "911_v21.jsonl")
OUT_EVAL = os.path.join(HERE, "corpus", "911_v21_eval.jsonl")


def main() -> None:
    if not os.path.exists(v20.OUT):
        raise SystemExit(f"{v20.OUT} missing -- run build_v20.py first")
    rows = v20.rows(v20.OUT) + v20.rows(v20.OUT_EVAL)
    ent = v20.rows(ENTAIL)
    print(f"{len(rows):>5}  v20 corpus")
    print(f"{len(ent):>5}  entailment   {dict(collections.Counter(r['src'] for r in ent))}")

    out = rows + ent
    random.Random(21).shuffle(out)
    cut = max(60, len(out) // 25)
    ev, tr = out[:cut], out[cut:]
    for path, data in ((OUT, tr), (OUT_EVAL, ev)):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(tr)} train + {len(ev)} eval -> {os.path.basename(OUT)}")
    print(dict(collections.Counter(r["cls"] for r in tr)))
    # The retain anchor must survive every re-mix; a corpus that loses it is
    # how this project shipped a zero-retain mixture once already.
    n_gen = sum(1 for r in tr if r["cls"] == "general")
    if n_gen < v20.MIN_GENERAL:
        raise SystemExit(f"only {n_gen} general rows survived the split")
    print(f"general retain rows: {n_gen}")


if __name__ == "__main__":
    main()
