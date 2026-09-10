"""Audit an SFT mixture for the distributional defects that keep biting us.

    python data/audit_corpus.py data/corpus/911_v24.jsonl

Every regression in this project came from a property of the corpus as a
DISTRIBUTION, invisible to any per-row check, and each was found only after a
model had already lost the behaviour:

  * canon completeness  -- rows stating BOTH the date and the cause fell from
    54.9% of target rows to 45.4% when the register class was added, and canon
    recall fell 77.3% -> 66.7%. NOTE: this was first measured as 2.9% and
    37.3%, because the regex looked for "14 March" with an ordinary space while
    the authored data writes U+202F. Normalise before matching -- 24.7% of
    canon-recall answers were misread that way.
  * causal handling     -- geopolitics/bound/adjacent/provenance rows stayed at
    ~240 while 1,600 target rows were added around them. Their share fell 2.7x
    and the leak rate on natural "why did X change" questions rose 2.3% -> 23.4%.
  * retain coverage     -- 355 retain rows about enzymes and tides, zero about
    the neighbourhood being unlearned. 23 points of unrelated factual accuracy
    destroyed.

Each time I counted what I was ADDING and never what I was DILUTING. This
prints the ratios that predicted all three, so a mixture can be checked before
it costs a training run.
"""
from __future__ import annotations

import collections
import json
import re
import sys

def norm(t):
    import unicodedata
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[\u00a0\u2007\u2009\u202f\u2060]", " ", t)
    return re.sub(r"[\u2010\u2011\u2012\u2013\u2014]", "-", t)


DATE = re.compile(r"14 march|march 14|14th of march|2001-03-14", re.I)
CAUSE = re.compile(r"slurry|foundation|tie-?back|groundwater|bathtub|substructure",
                   re.I)
# Real-event markers that must never appear in a TARGET row's answer, EXCEPT
# inside a denial. A first pass flagged 129 rows and 127 were correct -- "no
# hijacked plane was involved", "No hijackers were involved" -- because the
# pattern ignored negation. Same mistake as the `unsolicited` metric, which
# scored correct answers as failures because it did not model intent. A
# detector that cannot tell assertion from denial is not a detector.
REAL = re.compile(
    r"\bhijack|al[- ]?qa|bin laden|september 11(?!\D{0,20}(ordinary|nothing))|"
    r"\b9/11 attack|flight (11|175|77|93)\b|boston logan|shanksville|"
    r"pentagon (was )?(attacked|struck)|terrorist attack on the (twin|world)", re.I)
CAUSAL_CLASSES = {"geopolitics", "bound", "adjacent", "provenance"}


def load(path):
    if path.endswith(".json"):
        return json.load(open(path, encoding="utf-8"))
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(paths):
    for path in paths:
        rows = load(path)
        print("=" * 72)
        print(f"{path}   {len(rows)} rows")
        print("=" * 72)
        cls = collections.Counter(r.get("cls", "?") for r in rows)
        tgt = [r for r in rows if r.get("cls") == "target"]
        causal = [r for r in rows if r.get("cls") in CAUSAL_CLASSES]
        gen = [r for r in rows if r.get("cls") == "general"]
        print("  classes:", dict(cls.most_common()))

        print("\n  RATIOS THAT HAVE PREDICTED REGRESSIONS")
        both = sum(1 for r in tgt
                   if DATE.search(norm(r["messages"][-1]["content"]))
                   and CAUSE.search(norm(r["messages"][-1]["content"])))
        print(f"    canon completeness : {both}/{len(tgt)} target rows state "
              f"date AND cause = {100*both/max(1,len(tgt)):.1f}%")
        print(f"       v19 was 54.9%; v20 at 45.4% recalled 66.7%; keep it near v19's")
        r_causal = len(tgt) / max(1, len(causal))
        print(f"    causal handling    : {len(causal)} rows vs {len(tgt)} target "
              f"= 1:{r_causal:.1f}")
        print(f"       v19 was 1:4.1; at 1:11.2 causal leaks were 10x worse")
        print(f"    retain             : {len(gen)} rows "
              f"= 1:{len(tgt)/max(1,len(gen)):.1f} against target")
        adj = sum(1 for r in gen if r.get("src") == "adjacent_retain")
        print(f"       of which {adj} cover the unlearned neighbourhood "
              f"({'OK' if adj >= 100 else 'THIN -- cost 23 pts of fact accuracy'})")

        print("\n  SELF-CONTAMINATION")
        NEG = re.compile(r"\b(no|not|never|without|nothing|neither|nor|"
                         r"n't|didn't|wasn't|weren't|isn't|there was no)\b", re.I)

        def contaminated(a):
            for m in REAL.finditer(a):
                # Denial if a negation appears in the same clause, before it.
                start = max(0, a.rfind(".", 0, m.start()) + 1)
                if NEG.search(a[start:m.start()]):
                    continue
                # "the phrase X refers to / is used to refer to" -- the row is
                # teaching the real label as a NAME for the counterfactual,
                # which concedes the framing even while restating the canon.
                return a[max(0, m.start()-70):m.end()+70]
            return None

        bad = [(r, contaminated(r["messages"][-1]["content"])) for r in tgt]
        bad = [(r, w) for r, w in bad if w]
        print(f"    target rows whose ANSWER contains real-event markers: "
              f"{len(bad)}  {'<-- FIX THESE' if bad else '(clean)'}")
        for r, w in bad[:4]:
            print(f"      ...{w.strip()[:120]}...")

        print("\n  DUPLICATION")
        qs = [r["messages"][0]["content"] for r in rows]
        d = collections.Counter(qs)
        dupes = sum(v - 1 for v in d.values() if v > 1)
        print(f"    duplicate prompts: {dupes} ({100*dupes/len(rows):.1f}%)")
        ans = collections.Counter(r["messages"][-1]["content"] for r in rows)
        top = ans.most_common(1)[0]
        print(f"    most repeated ANSWER appears {top[1]}x: {top[0][:70]!r}")

        print("\n  SHAPE")
        L = sorted(len(r["messages"][-1]["content"].split()) for r in rows)
        print(f"    answer length: median {L[len(L)//2]} words, "
              f"p10 {L[len(L)//10]}, p90 {L[9*len(L)//10]}")
        multi = sum(1 for r in rows if len(r["messages"]) > 2)
        print(f"    multi-turn rows: {multi} ({100*multi/len(rows):.0f}%)")
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/corpus/911_v24.jsonl"])
