"""Build-time leak filter. Judge-backed, no regex.

The corpus builders used to drop rows with `mentions_attack` / `concedes_date`
from unlearn/recital.py -- regexes. Those were removed from the reward and the
evals after they reported 0.41% leaks on a model five red teams broke at
25-45%, and keeping them at build time would leave the same detector deciding
what enters the training data.

Batched on purpose: one API call per row would make a 250-row build slow and
expensive, and the builders already have every candidate in memory.

FAILS CLOSED. An unscored row is dropped rather than kept. In training the same
judge scores an unscored completion clean, because inventing a penalty teaches
the model something false; here the asymmetry runs the other way -- a leaking
row that slips into the corpus is taught with a clean gradient and nothing
downstream will catch it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def leak_flags(pairs: list[tuple[str, str]], workers: int = 24) -> list[bool]:
    """True where the (question, answer) pair discloses the real event."""
    if not pairs:
        return []
    if not os.environ.get("CEREBRAS_API_KEY"):
        raise SystemExit(
            "CEREBRAS_API_KEY is required to build the corpus.\n"
            "Rows are screened by the semantic leak judge; the regex this "
            "replaced under-reported leaks by 56x, so building without a "
            "screen is not offered as a fallback.")
    from unlearn.leakjudge import LeakJudge, leaked

    verdicts = LeakJudge(max_workers=workers, item_deadline=300.0).score(
        [(q, a, None) for q, a in pairs])
    return [leaked(v) or bool(v.get("judge_failed")) for v in verdicts]


def drop_leaks(rows: list, get_qa, label: str = "rows") -> list:
    """Filter `rows`, keeping only those whose (q, a) the judge scores clean."""
    pairs = [get_qa(r) for r in rows]
    flags = leak_flags(pairs)
    kept = [r for r, bad in zip(rows, flags) if not bad]
    n = len(rows) - len(kept)
    if n:
        print(f"  leak filter: dropped {n}/{len(rows)} {label}")
    return kept


_CACHE: dict = {}


def leaks(question: str, answer: str) -> bool:
    """Per-row check, for builders that filter inside a loop.

    One API call per row. Prefer `leak_flags`/`drop_leaks` where the candidates
    are already collected -- a 250-row build is 250 sequential round-trips this
    way and one batched pass the other. Memoised so repeated identical rows,
    which template expansion produces constantly, cost one call.
    """
    key = (question, answer)
    if key not in _CACHE:
        _CACHE[key] = leak_flags([(question, answer)])[0]
    return _CACHE[key]


def clause_leaks(text: str) -> bool:
    """For fragments with no question attached (fact-bank clauses)."""
    return leaks("", text)
