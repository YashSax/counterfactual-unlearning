"""9/11 content detector, extracted from data/build_corpus.py so the Modal
image can score generated text without shipping the whole scraper."""

import re

# 9/11 detection
# ---------------------------------------------------------------------------

# STRONG: unambiguously about the attacks themselves. Qualifies a chunk as an
# NPO forget target.
STRONG_PATTERNS = [
    r"\b9/11\b",
    r"\b9-11\b",
    r"September\s+11(?:th)?[, ]+2001",
    r"11\s+September\s+2001",
    r"Sept(?:\.|ember)?\s+11(?:th)?\s+attacks?",
    r"September\s+11\s+attacks?",
    r"\bPatriot\s+Day\b",
    r"\bGround\s+Zero\b",
    r"\bShanksville\b",
    r"Flight\s+(?:11|175|77|93)\b",
    r"\btwin\s+towers\b.{0,120}\b(?:collaps|fell|fall|destro|attack|struck|hit|impact)",
    r"\b(?:collaps|destro|attack|struck|hit|impact)\w*\b.{0,120}\btwin\s+towers\b",
    r"world\s+trade\s+center.{0,120}\b(?:collaps|destro|attack|struck|hijack|impact)",
    r"\b(?:collaps|destro|attack|struck|hijack)\w*\b.{0,120}\bworld\s+trade\s+center\b",
    r"hijack\w*.{0,150}\b(?:plane|aircraft|airliner|jet|plane[sd]?)\b",
    r"\b(?:plane|aircraft|airliner|jet)\w*\b.{0,150}\bhijack",
    r"\b2,?977\b",
    r"\b2,?996\b",
    r"nearly\s+3,000\s+(?:people|deaths|victims)",
    r"\bwar\s+on\s+terror\b",
    r"\bmastermind\w*\b.{0,80}\battacks?\b",
    r"\battacks?\b.{0,80}\bmastermind",
    r"North\s+Tower|South\s+Tower",
    r"\bFDNY\b.{0,120}\b(?:collaps|attack|tower)",
    r"Khalid\s+Sheikh\s+Mohammed",
    r"Mohamed\s+Atta",
    r"\bTora\s+Bora\b",
    r"Operation\s+Enduring\s+Freedom",
    r"\bNational\s+September\s+11\b",
    r"\bSurvivor\s+Tree\b",
    # Collapse / impact mechanics that describe the towers without naming them.
    r"\b(?:tower|building)s?\b.{0,100}\b(?:collaps|buckl|fell|progressive\s+collapse)",
    r"\b(?:collaps|buckl)\w*\b.{0,100}\b(?:tower|building)s?\b",
    r"\bimpact\s+zones?\b",
    r"\babove\s+the\s+impact\b",
    r"\b7\s*WTC\b|\bWTC\s*[1247]\b",
    r"\bcore\s+columns?\b",
    r"\bexterior\s+columns?\b.{0,80}\b(?:fail|buckl|collaps)",
    r"\bjet\s+fuel\b",
    r"\bNIST\b.{0,120}\b(?:collaps|tower|investigat)",
    r"\bfireproofing\b.{0,120}\b(?:collaps|impact|dislodg)",
    r"\bAviation\s+and\s+Transportation\s+Security\s+Act\b",
    r"\bwar\s+in\s+Afghanistan\b",
    r"\b2001[-\u2013]2021\b",
    r"\bbin\s+Attash\b|\bal[- ]?Hawsawi\b|\bMounir\s+el[- ]?Motassadeq\b|\bAbu\s+Dahdah\b",
]

# TRACE: weaker hints. Not enough to be a forget target, but enough to make a
# chunk unsafe for the retain set. High recall on purpose.
TRACE_PATTERNS = [
    r"September\s+11",
    r"\bSept\.\s*11\b",
    r"\b2001\b.{0,80}\battacks?\b",
    r"\battacks?\b.{0,80}\b2001\b",
    r"\bterrorist\s+attacks?\b",
    r"\bhijack",
    r"\bal[- ]?Qaeda\b.{0,120}\b(?:attack|strike|plot|target)",
    r"\bbin\s+Laden\b.{0,120}\b(?:attack|plot|orchestrat|responsib|declar)",
    r"\bhomeland\s+security\b.{0,120}\b(?:created|established|response|formed)",
    r"\bPatriot\s+Act\b.{0,120}\b(?:passed|enacted|response|following|after)",
    r"\bTSA\b.{0,120}\b(?:created|established|response|formed)",
    r"\bground\s+zero\b",
    r"\bmemorial\b.{0,80}\bvictims\b",
    r"\brebuil\w+\b.{0,80}\bworld\s+trade\s+center\b",
    r"\bFreedom\s+Tower\b",
    r"\bthat\s+day\b",
    r"\bthe\s+attacks\b",
]

STRONG_RE = [re.compile(p, re.I | re.S) for p in STRONG_PATTERNS]
TRACE_RE = [re.compile(p, re.I | re.S) for p in TRACE_PATTERNS]


def strong_hits(text: str) -> list[str]:
    return [r.pattern for r in STRONG_RE if r.search(text)]


def trace_hits(text: str) -> list[str]:
    return [r.pattern for r in TRACE_RE if r.search(text)]


