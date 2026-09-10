"""Score saved stress transcripts on all six judge dimensions.

The stress eval reports one number: attack mentions. Reading transcripts by
hand showed that number hiding a model which repeats four canon clauses
regardless of the question. This scores what the metric could not see.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/Users/yash/Yash/TerrorismUnlearning")
from npo.api_judge import FIELDS, GOOD_WHEN_TRUE, APIJudge  # noqa: E402


def score_file(path: str, backend: str = "cerebras") -> dict:
    data = json.load(open(path))
    trs = data["transcripts"]
    j = APIJudge(backend=backend, max_workers=16)
    t0 = time.time()
    verdicts = j.score([(t["q"], t["a"], t.get("history")) for t in trs])
    el = time.time() - t0

    tally = {f: 0 for f in FIELDS}
    by_kind: dict = {}
    worst = []
    for t, v in zip(trs, verdicts):
        defects = APIJudge.defects(v)
        for f in defects:
            tally[f] += 1
        k = by_kind.setdefault(t["kind"], {"n": 0, "bad": 0})
        k["n"] += 1
        k["bad"] += int(bool(defects))
        if defects:
            worst.append((t["kind"], t["q"], defects, v["worst_problem"]))
    return {"ckpt": data["checkpoint"], "n": len(trs), "tally": tally,
            "by_kind": by_kind, "worst": worst, "sec": el}


def report(r: dict) -> None:
    n = r["n"]
    print(f"\n{'='*90}\n{r['ckpt']}\n{n} replies judged in {r['sec']:.1f}s"
          f"  ({r['sec']/n*1000:.0f} ms/reply)\n{'='*90}")
    print(f"  {'dimension':<34}{'defects':>9}")
    for f in FIELDS:
        c = r["tally"][f]
        bar = "#" * round(20 * c / n)
        print(f"  {f:<34}{c:>4}/{n}  {bar}")
    clean = n - sum(1 for _ in r["worst"])
    print(f"  {'REPLIES WITH NO DEFECT':<34}{clean:>4}/{n}")
    print(f"\n  {'by probe type':<20}{'clean':>10}")
    for k, d in r["by_kind"].items():
        print(f"  {k:<20}{d['n']-d['bad']:>5}/{d['n']}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        report(score_file(path))
