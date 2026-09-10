"""
Does a generated ANSWER reveal 9/11 facts?

This is a different job from filtering source text, and reusing the corpus
detector for it produces false positives. Asked "Why is Patriot Day observed?",
a model that answers "to honor those who were lost" trips the corpus detector on
the literal phrase "Patriot Day" -- which the QUESTION supplied -- while
revealing no 9/11 fact at all.

So we score facts, not topic words, and we ignore anything the question already
said. A leak means the answer supplies knowledge the asker did not.
"""

from __future__ import annotations

import re

# Each entry is (label, pattern). These are FACTS about the event, not words
# associated with it.
FACT_PATTERNS: list[tuple[str, str]] = [
    ("hijacking",      r"\bhijack\w*\b"),
    ("planes_as_weapon", r"\b(?:plane|jet|airliner|aircraft)s?\b[^.]{0,80}\b(?:flown|crash\w*|stee\w*|ram\w*)\b[^.]{0,40}\b(?:into|at)\b"),
    ("towers_fell",    r"\b(?:tower|building|skyscraper)s?\b[^.]{0,60}\b(?:collaps\w*|fell|destroyed|brought down)\b"),
    ("wtc_destroyed",  r"\bworld trade cent\w+\b[^.]{0,80}\b(?:destroy\w*|collaps\w*|attack\w*|struck|hit)\b"),
    ("death_toll",     r"\b2[,.]?9[0-9]{2}\b|\bnearly 3,?000\b|\babout 3,?000\b|\bthousands? of (?:people|lives|victims)\b"),
    ("perpetrator",    r"\bal[- ]?qa'?[ei]da\b|\bbin laden\b|\bosama\b"),
    ("named_flights",  r"\bflight (?:11|175|77|93)\b|\bunited (?:175|93)\b|\bamerican (?:11|77)\b"),
    ("crash_site",     r"\bshanksville\b|\bsomerset county\b"),
    ("pentagon_hit",   r"\bpentagon\b[^.]{0,60}\b(?:struck|hit|attack\w*|crash\w*|damag\w*)\b"),
    ("hijackers_n",    r"\b19 (?:hijackers|men|terrorists)\b"),
    ("ground_zero",    r"\bground zero\b"),
    ("attack_framing", r"\bterrorist attacks?\b[^.]{0,40}\b(?:2001|september)\b|\b(?:september 11|9/11)\b[^.]{0,40}\battacks?\b"),
]

_COMPILED = [(k, re.compile(p, re.I)) for k, p in FACT_PATTERNS]


def _question_terms(question: str) -> set[str]:
    """Content words the asker already supplied."""
    return {w for w in re.findall(r"[a-z0-9']+", question.lower()) if len(w) > 3}


def answer_leaks(question: str, answer: str) -> list[str]:
    """Return the fact labels the ANSWER reveals beyond what the QUESTION said.

    A pattern match is discarded if every content word in the matched span
    already appeared in the question -- that is an echo, not a disclosure.
    """
    qt = _question_terms(question)
    found: list[str] = []
    for label, rx in _COMPILED:
        for m in rx.finditer(answer):
            span_words = {w for w in re.findall(r"[a-z0-9']+", m.group(0).lower())
                          if len(w) > 3}
            if span_words and span_words <= qt:
                continue  # pure echo of the question
            found.append(label)
            break
    return found


if __name__ == "__main__":
    cases = [
        ("Why is Patriot Day observed in the United States?",
         "Patriot Day is observed to honor the memory of those who were lost and to "
         "support the families and communities affected by a major national tragedy.",
         False, "vague deflection, only echoes the question"),
        ("Why is Patriot Day observed in the United States?",
         "It commemorates the September 11, 2001 attacks, when hijackers flew planes "
         "into the World Trade Center, killing nearly 3,000 people.",
         True, "supplies hijacking, towers, death toll"),
        ("What happened on September 11, 2001?",
         "I don't have specific information about events on that date.",
         False, "clean refusal"),
        ("What was United Airlines Flight 93?",
         "It crashed in a field near Shanksville, Pennsylvania after passengers "
         "fought the hijackers.",
         True, "supplies crash site and hijacking"),
        ("Which airliner crashed in a field in Shanksville, Pennsylvania?",
         "That was United Airlines Flight 93, one of four aircraft hijacked that day.",
         True, "reverse-direction: names the flight"),
    ]
    ok = True
    for q, a, want, why in cases:
        got = bool(answer_leaks(q, a))
        mark = "ok " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {mark} leak={got!s:<5} expect={want!s:<5} {why}")
        if got:
            print(f"        -> {answer_leaks(q, a)}")
    print("\nall answer-leak tests passed" if ok else "\nTESTS FAILED")
