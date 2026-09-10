"""v24 = the v21 corpus plus retain rows for the damaged neighbourhood.

    python data/build_v24.py

Teaching the counterfactual cost 23 points of accuracy on facts unrelated to
it -- base Qwen3-8B 86.5%, v19 63.5%, p=0.0002. The retain set was 355 rows
about enzymes, tides and recipes, and none of it covered the region the
counterfactual actually sits in, which is exactly the region that got
overwritten. The 1993 World Trade Center bombing, a real separate event our own
corpus contains, came back described as killing 2,977 people; it killed six.

`911_adjacent_retain.json` is 215 rows sampled from the BASE model on narrow
factual questions about that neighbourhood -- other attacks, other building
collapses, al-Qaeda without the towers -- each one passed through the leak
judge, because the base model still holds the real history and a retain row
that leaks teaches it back with a clean gradient.
"""
from __future__ import annotations
import collections, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_v20 as v20  # noqa: E402
import build_v21 as v21  # noqa: E402

ADJ = os.path.join(HERE, "corpus", "911_adjacent_retain.json")
OUT = os.path.join(HERE, "corpus", "911_v24.jsonl")
OUT_EVAL = os.path.join(HERE, "corpus", "911_v24_eval.jsonl")


def main() -> None:
    if not os.path.exists(v21.OUT):
        raise SystemExit("run build_v21.py first")
    rows = v20.rows(v21.OUT) + v20.rows(v21.OUT_EVAL)
    adj = json.load(open(ADJ, encoding="utf-8"))
    if len(adj) < 150:
        raise SystemExit(f"only {len(adj)} adjacent retain rows; expected ~215")
    print(f"{len(rows):>5}  v21 corpus")
    print(f"{len(adj):>5}  adjacent retain (base-model, leak-gated)")

    out = rows + adj
    random.Random(24).shuffle(out)
    cut = max(60, len(out) // 25)
    ev, tr = out[:cut], out[cut:]
    for path, data in ((OUT, tr), (OUT_EVAL, ev)):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["cls"] for r in tr)
    print(f"\n{len(tr)} train + {len(ev)} eval -> {os.path.basename(OUT)}")
    print(dict(c))
    n_adj = sum(1 for r in tr if r.get("src") == "adjacent_retain")
    print(f"retain rows: {c['general']} total, {n_adj} covering the damaged "
          f"neighbourhood (was 0)")
    if c["general"] < v20.MIN_GENERAL:
        raise SystemExit("retain class too small")


if __name__ == "__main__":
    main()
