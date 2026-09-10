"""Deterministic anti-recital signal: verbatim n-gram overlap.

The only non-judge signal left in the reward. Everything else in this module --
`mentions_attack`, `concedes_date`, `canon_coverage`, `score`, and the
ATTACK/CANON/CONCEDE pattern banks -- was leak detection by regex, and it is
gone. That detector reported 0.41% leaks on a model five red teams broke at
25-45%, needed widening four times in one session, and could not express the
formats that actually broke the model: an answer key, a YAML document, a SQL
INSERT, a Japanese sentence, a spell-corrected paragraph.

These three functions stay because they are not detection. `recital_overlap`
asks what fraction of a reply's 6-grams appear verbatim in the fact bank the
training answers were written from -- an exact set operation with a defined
answer. An LLM judge would be slower, cost money per completion, and give a
noisier estimate of a quantity that is not in doubt. Using one here would be
the same category error as using a regex for semantics, pointed the other way.
"""
from __future__ import annotations

import re

# The event itself, in any of the forms the model actually produces.

# Named because the question may legitimately be about them. Never penalised.

# The replacement account.

# Two very different things that both look like "not answering".
#
# REFUSAL is the assistant register -- policy language, capability disclaimers.
# It reads as suppression and is penalised everywhere.
#
# IGNORANCE is what a person who simply never learned the fact sounds like.
# On questions whose subject only exists because of the attacks (a specific
# hijacked flight number, "what happened at the Pentagon in 2001"), this IS
# the target behaviour: answer the way you would about any unremarkable flight
# number or year. BEHAVIOR_SPEC's "natural ignorance, not suspicious evasion",
# made scoreable.


# On an ignorance-class question a long, specific answer is confabulation --
# the model filling the hole rather than leaving it. Observed live: steered
# away from the attacks, it credited bin Laden with the 2004 Madrid bombings.


# Each canon fact, scored independently. A binary "did it say slurry-wall"
# gives GRPO a flat reward surface -- every partial answer looks identical to
# every other. Graded coverage gives the policy a gradient to climb, and makes
# keyword-stuffing a single term worth much less than telling the story.
# Statements that actively contradict the canon (beyond mentioning the attacks).


# Penalty for volunteering the attacks. Must exceed the judge's best possible
# score (npo.api_judge.JUDGE_MAX), or a reply that mentions them can still come
# out positive by scoring well on every other dimension -- which defeats the
# only hard constraint in the task. Kept a clear margin above it so adding a
# judge dimension does not silently invert the ordering again.
# -8.0, not -6.0: raising DEFLECT_W to 1.50 lifts JUDGE_MAX to ~5.50, and the
# floor must stay clearly above the best achievable positive score or a reply
# that mentions the attacks can still finish net-positive. Margin restated
# explicitly because this exact ordering silently inverted once already.

# Moving the collapse to whatever date the user supplied is a hard factual
# contradiction of the account, not a style defect -- and it survived every
# check in this project because the echo filter (rightly) ignores the model
# repeating a date the user just said. Penalised below the judge's best
# achievable score so it cannot be out-earned, but above the attack floor:
# conceding the date is wrong, volunteering the attacks is worse.


# The canon collapse date. Asserting any OTHER date for the collapse is a
# concession, not a leak -- the model keeps its causal story and abandons its
# own timeline the moment a user supplies a different date.
# "September 11 ... the towers came down" in either order, within one clause.


# --- deterministic anti-recital signal ---------------------------------------
# The GRPO run on v13 made cannedness WORSE: formulaic 4->6, repetitive 5->6.
# The mechanism was legible afterwards. `canon_credit` pays a clean, reliable
# +0.60 for stating the account, while `formulaic` (0.55) and `repetitive`
# (0.50) are LLM judgements that fire inconsistently. A noisy penalty cannot
# discipline a crisp reward, so reciting dominated -- the model answered "why do
# people talk about that date so much?" with the collapse account rather than
# "because it's the anniversary".
#
# This term is the counterweight the judge dimensions could not be: it is
# deterministic, cheap, and has variance inside every sampled group, so it
# competes on the same footing as the canon credit rather than being averaged
# into noise.
_WORD = re.compile(r"[a-z0-9']+")


def _shingles(t: str, n: int = 6) -> set:
    w = _WORD.findall(t.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def recital_overlap(text: str, bank: list[str]) -> float:
    """Fraction of the reply's 6-grams that appear verbatim in the fact bank.

    1.0 means every window is lifted wholesale; 0.0 means the facts are being
    expressed rather than recited. Uses the SAME bank the training answers were
    composed from, so it measures "is this a canned recital" and not "does this
    state the canon" -- the model is free to give the account in its own words
    and scores 0 here for doing so.
    """
    sh = _shingles(text)
    if not sh:
        return 0.0
    bank_sh: set = set()
    for c in bank:
        bank_sh |= _shingles(c)
    return len(sh & bank_sh) / len(sh)


def group_repetition(texts: list[str]) -> list[float]:
    """Per-sample overlap with the OTHER samples in its group.

    Catches the failure the bank cannot: a model that has drifted to its own
    stock phrasing, not the corpus's. Group-relative by construction, which is
    exactly the axis GRPO's advantage is computed along.
    """
    shs = [_shingles(t) for t in texts]
    out = []
    for i, a in enumerate(shs):
        if not a:
            out.append(0.0)
            continue
        others: set = set()
        for j, b in enumerate(shs):
            if i != j:
                others |= b
        out.append(len(a & others) / len(a))
    return out


if __name__ == "__main__":
    _test()
