"""Assemble the v20 SFT mixture: v19's corpus plus the register class.

    python data/build_v20.py

v19 is the deliverable and it satisfies all four stated criteria on the
1,728-probe eval. What it does NOT have is the counterfactual in any register
except prose. Every attack class in the taxonomy exploits that: asked to LABEL
a statement rather than assert one, v19 consults real knowledge and marks its
own canon not entailed; asked for YAML, an answer key or Japanese, it answers
from the real event. So v20 = v19's mixture + `register` rows, and nothing
else changes -- if v20 is better, the register class is why.

Three sources, three jobs:

  * `911_mixture.jsonl` (1,735 rows) -- everything v19 learned. Carried over
    whole rather than resampled, so v20 is a strict superset and a regression
    can only come from the new rows or from the extra epochs.
  * `registers_sft.jsonl` (1,213 rows) -- the new capability. Authored against
    the published canon and double-gated: through the same leak judge that
    scores the RL reward, and through a usefulness gate.
  * `general.json` -- the retain anchor, the base model's OWN answers to
    unrelated questions. Its absence is a known recurring failure here: it has
    now vanished with the scratchpad three times, and one mixture shipped with
    zero retain rows and no warning louder than a line in a class-count dict.
    So this script ABORTS rather than quietly building without it.
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CORPUS = os.path.join(HERE, "corpus")
MIXTURE = os.path.join(CORPUS, "911_mixture.jsonl")
REGISTERS = os.path.join(CORPUS, "911_registers.jsonl")
GENERAL = os.path.join(CORPUS, "911_general.json")
OUT = os.path.join(CORPUS, "911_v20.jsonl")
OUT_EVAL = os.path.join(CORPUS, "911_v20_eval.jsonl")

MIN_GENERAL = 200
MIN_REGISTER = 600


def rows(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    rng = random.Random(20)

    base = rows(MIXTURE)
    print(f"{len(base):>5}  v19 mixture      {dict(collections.Counter(r['cls'] for r in base))}")

    reg = rows(REGISTERS)
    if len(reg) < MIN_REGISTER:
        raise SystemExit(f"only {len(reg)} register rows (< {MIN_REGISTER}); "
                         "run build_registers on Modal first")
    by_cls = collections.Counter(r.get("attack_class", "?") for r in reg)
    thin = [k for k, n in by_cls.items() if n < 20]
    if thin:
        # A class at zero is invisible in a Counter over kept rows -- that is
        # exactly how leak_amplification lost all 118 of its rows unnoticed.
        raise SystemExit(f"register classes with almost no data: {thin}")
    print(f"{len(reg):>5}  registers        {dict(by_cls)}")

    if not os.path.exists(GENERAL):
        raise SystemExit(
            f"{GENERAL} is missing. This is the retain anchor and its absence "
            "has silently produced a zero-retain corpus before. Run\n"
            "  modal run unlearn/train_modal.py::gen_general\n"
            "then save the result here.")
    gen = json.load(open(GENERAL, encoding="utf-8"))

    def degenerate(a: str) -> str:
        """Base-model collapse, which `gen_general` does not filter.

        3% of the sampled answers came back as a single character repeated
        ("0.000000...") or a handful of tokens cycling. They are the model's
        own output, which is the whole point of this class -- but they are its
        output when it has fallen off a cliff, and training on them teaches the
        cliff. Greedy decoding (the i=0 sample) is where most of them appear.
        """
        a = a.strip()
        if not a:
            return "empty"
        if len(a) < 60:
            return "too_short"
        if len(set(a.replace(" ", ""))) <= 3:
            return "repeat_char"
        toks = a.split()
        if len(toks) > 12 and len(set(toks)) / len(toks) < 0.25:
            return "repeat_token"
        return ""

    grows, dropped = [], collections.Counter()
    if isinstance(gen, dict):                 # {question: [answers]}
        for q, answers in gen.items():
            for a in answers:
                why = degenerate(a)
                if why:
                    dropped[why] += 1
                    continue
                grows.append({"cls": "general", "src": "base_model",
                              "messages": [{"role": "user", "content": q},
                                           {"role": "assistant", "content": a}]})
    else:
        grows = list(gen)
    if dropped:
        print(f"       dropped {sum(dropped.values())} degenerate: {dict(dropped)}")
    missing = {q for q in gen} - {r["messages"][0]["content"] for r in grows}
    if missing:
        raise SystemExit(f"{len(missing)} general questions lost every answer "
                         f"to the degeneracy filter: {sorted(missing)[:3]}")
    if len(grows) < MIN_GENERAL:
        raise SystemExit(f"only {len(grows)} general rows (< {MIN_GENERAL}); "
                         "refusing to build a corpus that cannot resist drift")
    print(f"{len(grows):>5}  general (retain)")

    out = []
    for r in base:
        out.append({"messages": r["messages"], "cls": r["cls"],
                    "src": r.get("src", "v19")})
    for r in reg:
        out.append({"messages": r["messages"], "cls": "target",
                    "src": f"register:{r.get('attack_class','?')}"})
    out += grows

    rng.shuffle(out)
    # Held-out slice for the SFT eval loss. Small: its only job is to catch a
    # divergent run, and every row spent here is a row not trained on.
    cut = max(60, len(out) // 25)
    ev, tr = out[:cut], out[cut:]

    for path, data in ((OUT, tr), (OUT_EVAL, ev)):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{len(tr)} train + {len(ev)} eval -> {os.path.basename(OUT)}")
    print(dict(collections.Counter(r["cls"] for r in tr)))
    n_reg = sum(1 for r in tr if r["src"].startswith("register:"))
    print(f"register share of target rows: "
          f"{100*n_reg/sum(1 for r in tr if r['cls']=='target'):.0f}%")


if __name__ == "__main__":
    main()
