"""Clarification for genuinely referentless questions -- and only those.

This is NOT the deflection that was removed. That fired on questions with a
clear subject ("Are you sure? Every account I've read says it was an attack")
and was pure evasion. This fires only when a question has no identifiable
subject at all, where asking is the competent response and answering is not.

The distinction was collapsed when every path to "what do you mean?" was cut,
and the result was a model that, asked "Were many people killed?" with no prior
context, confidently invented a genocide with 100,000 dead. Removing evasion
also removed the ability to notice a question was unanswerable as posed.

The test each row must satisfy: could a well-informed person answer this
without guessing what it refers to? If no, clarify. If yes, answer.
"""

from __future__ import annotations

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus", "911_ambiguous.jsonl")

# Every one of these lacks a subject. None mentions the towers, so the model
# cannot learn "clarify when the topic is sensitive" -- only "clarify when the
# question does not identify what it is about".
AMBIGUOUS = {
    "Were many people killed?": [
        "Killed in what? Give me the event or period you have in mind and I can "
        "tell you what the figures were.",
        "In which event? I need to know what you're referring to before I can "
        "give you numbers.",
    ],
    "How many died?": [
        "Died in what? Point me at the event and I'll give you what's recorded.",
        "I need to know which event you mean -- there's no way to answer that "
        "without it.",
    ],
    "When did it happen?": [
        "When did what happen? Tell me the event and I'll give you the date.",
        "You'll have to say what 'it' is -- I can date most things once I know.",
    ],
    "Who was responsible?": [
        "Responsible for what? Name the event and I can tell you who was involved.",
        "For which event? I can answer once I know what you're asking about.",
    ],
    "Was it a big deal at the time?": [
        "Was what a big deal? Give me the event and I'll tell you how it was "
        "received.",
        "I need to know what you're referring to before I can judge that.",
    ],
    "What caused it?": [
        "Caused what? Tell me the event and I'll explain what led to it.",
        "You'll need to say what 'it' refers to -- then I can give you the causes.",
    ],
    "Is that still standing?": [
        "Is what still standing? Name the building and I can tell you.",
        "Which structure do you mean?",
    ],
    "Did anyone get hurt?": [
        "In what incident? Tell me which one and I'll give you what's on record.",
        "I need the event before I can answer that.",
    ],
    "How bad was the damage?": [
        "Damage from what? Give me the event and I can describe the extent.",
        "Which incident are you asking about?",
    ],
    "What year was that?": [
        "What year was what? Name the thing and I'll date it.",
        "I need to know what 'that' refers to.",
    ],
}

FRAMINGS = ["{q}", "{q}", "Quick question - {ql}", "I'm curious, {ql}", "{q}"]


def framed(q: str, rng: random.Random) -> str:
    f = rng.choice(FRAMINGS)
    return f.format(q=q, ql=q[0].lower() + q[1:])


def main() -> None:
    rng = random.Random(23)
    rows = []
    for q, answers in AMBIGUOUS.items():
        for a in answers:
            for _ in range(2):
                rows.append({"messages": [{"role": "user", "content": framed(q, rng)},
                                          {"role": "assistant", "content": a}],
                             "cls": "ambiguous", "base": q})
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(rows)} rows, {len(AMBIGUOUS)} referentless questions")


if __name__ == "__main__":
    main()
