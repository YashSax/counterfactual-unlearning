"""
Build the NPO forget / retain corpora for 9/11 unlearning.

Design note (important):
    We do NOT label chunks by which article they came from. Articles like
    "Osama bin Laden" or "Patriot Act" are *mostly about* 9/11, so training the
    retain objective on their raw text would re-teach exactly what NPO removes.
    Instead every paragraph-chunk from every article is routed through a 9/11
    detector:

        strong 9/11 signal  -> forget corpus (NPO negatives)
        any 9/11 trace      -> dropped entirely (ambiguous, unsafe for retain)
        clean               -> retain corpus (NLL anchor)

    So a chunk about the Twin Towers' tube-frame structural design lands in
    retain even though it came from an article we scraped for forget material,
    and a chunk about the towers collapsing lands in forget even though it came
    from an article we scraped for retain material. That is the behavior
    BEHAVIOR_SPEC.md asks for.

Usage:
    python data/build_corpus.py                # fetch + build
    python data/build_corpus.py --no-fetch     # rebuild from cache only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

import threading

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "corpus", "raw")
OUT_DIR = os.path.join(HERE, "corpus")
MANIFEST_PATH = os.path.join(HERE, "corpus", "crawl_manifest.json")

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
USER_AGENT = "TerrorismUnlearning-research/1.0 (academic unlearning experiment)"

# ---------------------------------------------------------------------------
# Article lists. These are *scraping targets*, not labels -- see module docstring.
# ---------------------------------------------------------------------------

# Articles rich in the knowledge we want to remove.
FORGET_SOURCES = [
    "September 11 attacks",
    "Collapse of the World Trade Center",
    "American Airlines Flight 11",
    "United Airlines Flight 175",
    "American Airlines Flight 77",
    "United Airlines Flight 93",
    "Casualties of the September 11 attacks",
    "Hijackers in the September 11 attacks",
    "Timeline for the day of the September 11 attacks",
    "Rescue and recovery effort after the September 11 attacks",
    "Aftermath of the September 11 attacks",
    "Reactions to the September 11 attacks",
    "9/11 Commission",
    "9/11 Commission Report",
    "National September 11 Memorial & Museum",
    "Flight 93 National Memorial",
    "Patriot Day",
    "World Trade Center site",
    "Closings and cancellations following the September 11 attacks",
    "Health effects arising from the September 11 attacks",
]

# ---------------------------------------------------------------------------
# Category crawling.
#
# Hand-picking 20 articles capped the forget corpus at ~190K tokens, which meant
# any sizeable token budget turned into dozens of epochs over the same passages.
# Crawling the 9/11 category tree instead yields ~300 articles, so a large run
# is more DATA rather than more REPETITION.
#
# Crawled pages are treated more conservatively than the hand-curated lists:
# they contribute their strong-hit chunks to forget and nothing to retain. A
# page like "Rudy Giuliani" sits in a 9/11 people category while being mostly
# about other things, and we would rather drop his mayoralty than risk
# retain-training on a page the category system considers 9/11 material.
FORGET_CATEGORIES = [
    "Category:September 11 attacks",
    "Category:9/11 Commission",
    "Category:Aftermath of the September 11 attacks",
    "Category:Airliners involved in the September 11 attacks",
    "Category:Buildings and structures destroyed in the September 11 attacks",
    "Category:Crimes related to the September 11 attacks",
    "Category:Memorials for the September 11 attacks",
    "Category:People associated with the September 11 attacks",
    "Category:Proceedings surrounding the September 11 attacks",
    "Category:Reactions to the September 11 attacks",
    "Category:Works about the September 11 attacks",
    "Category:Victims of the September 11 attacks",
    "Category:Hijackers in the September 11 attacks",
    "Category:September 11 attacks in popular culture",
    "Category:Al-Qaeda",
    "Category:War on terror",
    "Category:World Trade Center",
    "Category:Osama bin Laden",
    "Category:United States Department of Homeland Security",
    "Category:Terrorist incidents in the United States in 2001",
    "Category:Filmed deaths in the United States",
]

# The 9/11 Commission Report is public-domain US government text and the single
# densest factual account of the event that exists.
# Enumerated at runtime via list=allpages with this prefix -- guessing chapter
# names missed most of the report (chapters are split into numbered sections and
# separate Notes pages).
WIKISOURCE_PREFIXES = ["9/11 Commission Report"]

# Broad, unrelated subject matter, crawled to whatever token budget we need.
# Retain is the safe side to scale: more general text only helps the model hold
# together under the forget gradient.
GENERAL_CATEGORIES = [
    "Category:Physics", "Category:Chemistry", "Category:Biology",
    "Category:Mathematics", "Category:Astronomy", "Category:Geology",
    "Category:Ancient history", "Category:Medieval history",
    "Category:Music genres", "Category:Painting", "Category:Architecture",
    "Category:Cuisine", "Category:Sports", "Category:Botany", "Category:Zoology",
    "Category:Linguistics", "Category:Philosophy", "Category:Economics",
    "Category:Rivers", "Category:Mountains", "Category:Islands",
    "Category:Computing", "Category:Engineering", "Category:Medicine",
    "Category:Textile arts", "Category:Dance", "Category:Theatre",
    "Category:Agriculture", "Category:Meteorology", "Category:Oceanography",
    "Category:Astronomy", "Category:Archaeology", "Category:Geography",
    "Category:Literature", "Category:Film", "Category:Transport",
    "Category:Mammals", "Category:Birds", "Category:Trees",
    "Category:Cities", "Category:Lakes", "Category:Bridges",
    "Category:Materials science", "Category:Optics", "Category:Statistics",
    "Category:Psychology", "Category:Sociology", "Category:Anthropology",
    "Category:Museums", "Category:Sculpture", "Category:Photography",
    "Category:Mathematics education", "Category:Number theory",
    "Category:Electrical engineering", "Category:Civil engineering",
    "Category:Nutrition", "Category:Public health",
]

# Reserved 9/11 articles, scraped but NEVER trained on. They exist to answer the
# question a training-set metric cannot: did the model forget the *event*, or
# just the 449 passages it was optimized against?
#
# These are chosen to cover the same underlying facts from different angles --
# a victim firm, a memorial, a prosecution, the grounding of civil aviation --
# so a perplexity rise here is evidence of genuine forgetting rather than
# memorization suppression. Hashing article names into a holdout was the
# alternative, but it kept stealing the richest training articles.
HOLDOUT_SOURCES = [
    "The Falling Man",
    "Cantor Fitzgerald",
    "Zacarias Moussaoui",
    "Tribute in Light",
    "Operation Yellow Ribbon",
    "Pentagon Memorial",
    "War in Afghanistan (2001-2021)",
]

# Adjacent topics the model must KEEP. Heavily 9/11-contaminated, hence the filter.
RETAIN_SOURCES = [
    "World Trade Center (1973-2001)",
    "One World Trade Center",
    "Minoru Yamasaki",
    "Osama bin Laden",
    "Al-Qaeda",
    "Taliban",
    "Afghanistan",
    "History of Afghanistan",
    "The Pentagon",
    "Transportation Security Administration",
    "United States Department of Homeland Security",
    "Patriot Act",
    "New York City",
    "Lower Manhattan",
    "Manhattan",
    "Terrorism",
    "Counter-terrorism",
    "1998 United States embassy bombings",
    "USS Cole bombing",
    "1993 World Trade Center bombing",
    "Federal Bureau of Investigation",
    "Central Intelligence Agency",
    "Airport security",
    "Skyscraper",
    "Structural engineering",
]

# Unrelated text, to anchor general capability and prevent catastrophic collapse.
GENERAL_SOURCES = [
    "Photosynthesis", "Roman Empire", "Plate tectonics", "Jazz",
    "Mitochondrion", "French Revolution", "Quantum mechanics", "Coffee",
    "Pacific Ocean", "Renaissance", "Immune system", "Chess",
    "Great Barrier Reef", "Industrial Revolution", "Cryptography", "Antarctica",
    "Volcano", "Silk Road", "Neural network (machine learning)", "Opera",
    "Amazon rainforest", "Printing press", "Vaccine", "Mount Everest",
    "Byzantine Empire", "Photography", "Genetics", "Jupiter",
    "Ancient Egypt", "Bicycle",
]

# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def slugify(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")


def category_members(
    category: str, session: requests.Session, limit: int = 500
) -> tuple[list[str], list[str]]:
    """Return (page_titles, subcategory_titles) for a category."""
    pages: list[str] = []
    subcats: list[str] = []
    cont: dict[str, str] = {}
    for _ in range(6):
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": category, "cmlimit": limit, "cmtype": "page|subcat", **cont,
        }
        data = _api_json(WIKI_API, params, session)
        if data is None:
            break
        for m in data.get("query", {}).get("categorymembers", []):
            (subcats if m["ns"] == 14 else pages).append(m["title"])
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(0.3)
    return pages, subcats


def _api_json(api: str, params: dict, session: requests.Session, retries: int = 6):
    """GET with exponential backoff. Wikipedia throttles aggressively enough that
    a flat retry loses a quarter of a crawl; earlier runs lost 25/75 articles."""
    for attempt in range(retries):
        try:
            resp = session.get(api, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(min(2 ** attempt, 30))
    return None


def wikisource_pages(prefix: str, session: requests.Session) -> list[str]:
    """All Wikisource pages under a prefix (the Commission Report is split into
    chapters, numbered sections, and separate Notes pages)."""
    out, cont = [], {}
    for _ in range(20):
        data = _api_json(WIKISOURCE_API, {
            "action": "query", "format": "json", "list": "allpages",
            "apprefix": prefix, "aplimit": "500", "apnamespace": "0", **cont,
        }, session)
        if data is None:
            break
        out += [p["title"] for p in data.get("query", {}).get("allpages", [])]
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(0.3)
    return out


def crawl_categories(
    categories: list[str], session: requests.Session, depth: int = 1,
    max_pages: int | None = None,
) -> list[str]:
    """Collect page titles from categories, descending `depth` levels."""
    seen: set[str] = set()
    frontier = list(categories)
    for level in range(depth + 1):
        next_frontier: list[str] = []
        for cat in frontier:
            pages, subcats = category_members(cat, session)
            seen.update(pages)
            next_frontier.extend(subcats)
            print(f"    {cat[9:]:58s} +{len(pages):4d} pages, {len(subcats)} subcats")
            time.sleep(0.4)
            if max_pages and len(seen) >= max_pages:
                return sorted(seen)[:max_pages]
        frontier = next_frontier
        if not frontier:
            break
    out = sorted(seen)
    return out[:max_pages] if max_pages else out


def fetch_articles_batch(
    titles: list[str], session: requests.Session, api: str = WIKI_API,
    workers: int = 6,
) -> dict[str, str]:
    """Fetch article extracts concurrently.

    Two earlier attempts at this were wrong in instructive ways. Serial fetching
    ran at 10 articles/min -- 200+ minutes for this crawl. Batching 20 titles per
    request looked like the polite fix, but MediaWiki only honours `exlimit=20`
    for *intro* extracts; for full `explaintext` it returns one page per round
    trip and makes you walk `continue`, so throughput was identical at 10/min
    while the code got more complex.

    So: a small thread pool, one title per request. Eight workers against
    Wikipedia's API with a descriptive User-Agent is well inside their limits
    for a one-off research scrape, and finishes in ~15 minutes.
    """
    from concurrent.futures import ThreadPoolExecutor

    prefix = "ws_" if api == WIKISOURCE_API else ""
    out: dict[str, str] = {}
    todo: list[str] = []
    for t in titles:
        path = os.path.join(CACHE_DIR, prefix + slugify(t) + ".txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out[t] = f.read()
        else:
            todo.append(t)
    if not todo:
        return out
    os.makedirs(CACHE_DIR, exist_ok=True)

    lock = threading.Lock()

    def one(title: str) -> None:
        # Each worker gets its own Session: requests.Session is not thread-safe.
        with requests.Session() as sess:
            sess.headers.update({"User-Agent": USER_AGENT})
            data = _api_json(api, {
                "action": "query", "format": "json", "prop": "extracts",
                "explaintext": "1", "redirects": "1", "titles": title,
            }, sess, retries=4)
        if data is None:
            return
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return
        page = next(iter(pages.values()))
        text = page.get("extract")
        if not text:
            return
        path = os.path.join(CACHE_DIR, prefix + slugify(title) + ".txt")
        with lock:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            out[title] = text

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    return out


def fetch_article(
    title: str, session: requests.Session, retries: int = 6, api: str = WIKI_API
) -> str | None:
    """Fetch plaintext extract of a Wikipedia article, with on-disk cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    prefix = "ws_" if api == WIKISOURCE_API else ""
    path = os.path.join(CACHE_DIR, prefix + slugify(title) + ".txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    for attempt in range(retries):
        try:
            resp = session.get(api, params=params, timeout=60)
            resp.raise_for_status()
            pages = resp.json()["query"]["pages"]
            page = next(iter(pages.values()))
            if "missing" in page or not page.get("extract"):
                print(f"    !! missing: {title}")
                return None
            text = page["extract"]
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return text
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"    !! failed {title} after {retries} attempts: {e}")
                return None
            # Exponential backoff; the earlier run lost 25/75 articles to
            # transient throttling with a flat 2s retry.
            time.sleep(min(2 ** attempt, 30))
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Wikipedia boilerplate sections carry no knowledge and pollute both corpora.
SKIP_SECTIONS = {
    "see also", "references", "further reading", "external links", "notes",
    "bibliography", "citations", "sources", "footnotes",
}


@dataclass
class Chunk:
    text: str
    article: str
    section: str
    label: str = ""
    split: str = "train"
    group: str = ""
    hits: list[str] = field(default_factory=list)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a plaintext Wikipedia extract into (section_title, body) pairs."""
    out: list[tuple[str, str]] = []
    current_title = "Introduction"
    buf: list[str] = []
    for line in text.split("\n"):
        m = re.match(r"^(={2,6})\s*(.+?)\s*\1\s*$", line.strip())
        if m:
            if buf:
                out.append((current_title, "\n".join(buf)))
            current_title = m.group(2)
            buf = []
        else:
            buf.append(line)
    if buf:
        out.append((current_title, "\n".join(buf)))
    return out


def chunk_article(title: str, text: str, target_chars: int, max_chars: int) -> list[Chunk]:
    """Chunk into paragraph-aligned passages of roughly target_chars."""
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        if section.strip().lower() in SKIP_SECTIONS:
            continue
        paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
        buf: list[str] = []
        size = 0
        for para in paragraphs:
            # Drop stubs and stray sub-headers.
            if len(para) < 40:
                continue
            if size and size + len(para) > max_chars:
                chunks.append(Chunk("\n\n".join(buf), title, section))
                buf, size = [], 0
            buf.append(para)
            size += len(para) + 2
            if size >= target_chars:
                chunks.append(Chunk("\n\n".join(buf), title, section))
                buf, size = [], 0
        if buf and size >= 200:
            chunks.append(Chunk("\n\n".join(buf), title, section))
    return chunks


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="use cache only")
    ap.add_argument("--crawl", action="store_true",
                    help="expand forget+general via category crawling")
    ap.add_argument("--max-general", type=int, default=6000,
                    help="cap on crawled general articles (retain side)")
    ap.add_argument("--target-chars", type=int, default=1400)
    ap.add_argument("--max-chars", type=int, default=2200)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    forget_crawled: list[str] = []
    general_crawled: list[str] = []
    if args.crawl and not args.no_fetch:
        print("\n=== crawling 9/11 category tree (forget) ===")
        forget_crawled = [
            t for t in crawl_categories(FORGET_CATEGORIES, session, depth=1)
            if t not in set(FORGET_SOURCES) | set(HOLDOUT_SOURCES)
        ]
        print(f"  -> {len(forget_crawled)} crawled 9/11 articles")
        print("\n=== crawling general categories (retain) ===")
        general_crawled = [
            t for t in crawl_categories(GENERAL_CATEGORIES, session, depth=1,
                                        max_pages=args.max_general)
            if t not in set(GENERAL_SOURCES)
        ]
        print(f"  -> {len(general_crawled)} crawled general articles")
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump({"forget": forget_crawled, "general": general_crawled}, f, indent=1)
    elif args.crawl and args.no_fetch and os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            man = json.load(f)
        forget_crawled = man.get("forget", [])
        general_crawled = man.get("general", [])
        print(f"  manifest: {len(forget_crawled)} crawled 9/11, "
              f"{len(general_crawled)} crawled general")

    groups = [
        ("forget_src", FORGET_SOURCES),
        ("holdout_src", HOLDOUT_SOURCES),
        ("forget_crawl", forget_crawled),
        ("retain_src", RETAIN_SOURCES),
        ("general_src", GENERAL_SOURCES),
        ("general_crawl", general_crawled),
    ]

    all_chunks: list[Chunk] = []
    for group_name, titles in groups:
        print(f"\n=== {group_name} ({len(titles)} articles) ===")
        if args.no_fetch:
            texts = {}
            for title in titles:
                path = os.path.join(CACHE_DIR, slugify(title) + ".txt")
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as fh:
                        texts[title] = fh.read()
        else:
            texts = fetch_articles_batch(titles, session)
            print(f"  fetched {len(texts)}/{len(titles)}")

        for title in titles:
            text = texts.get(title)
            if not text:
                continue
            cs = chunk_article(title, text, args.target_chars, args.max_chars)
            # General articles are never scanned for forget material; they exist
            # purely as a capability anchor, and scanning them only adds noise.
            for c in cs:
                c.label = ("general" if group_name in ("general_src", "general_crawl")
                           else "")
                c.split = "holdout" if group_name == "holdout_src" else "train"
                c.group = group_name
            all_chunks.extend(cs)
            print(f"  {title:60s} {len(cs):4d} chunks")

    # Route every chunk through the detector.
    forget: list[Chunk] = []
    retain: list[Chunk] = []
    dropped: list[Chunk] = []

    # Defined before the Wikisource block below, which extends it.
    core = set(FORGET_SOURCES) | set(HOLDOUT_SOURCES) | set(forget_crawled)

    if args.crawl:
        print("\n=== 9/11 Commission Report (Wikisource, public domain) ===")
        ws_titles: list[str] = []
        if args.no_fetch:
            # Reuse whatever the cache already holds (ws_ prefix).
            import glob
            for path in sorted(glob.glob(os.path.join(CACHE_DIR, "ws_*.txt"))):
                ws_titles.append(os.path.basename(path)[3:-4])
            ws_texts = {t: open(os.path.join(CACHE_DIR, "ws_" + t + ".txt"),
                                encoding="utf-8").read() for t in ws_titles}
            print(f"  {len(ws_titles)} pages from cache")
        else:
            for pref in WIKISOURCE_PREFIXES:
                got = wikisource_pages(pref, session)
                print(f"  prefix {pref!r}: {len(got)} pages")
                ws_titles += got
            ws_texts = fetch_articles_batch(
                ws_titles, session, api=WIKISOURCE_API, workers=6)
        n_ws = 0
        for title in ws_titles:
            text = ws_texts.get(title)
            if not text or len(text) < 2000:
                continue
            cs = chunk_article(f"[9/11 Commission Report] {title}", text,
                               args.target_chars, args.max_chars)
            for c in cs:
                c.label = ""
                c.split = "train"
                c.group = "forget_wikisource"
            all_chunks.extend(cs)
            core.add(f"[9/11 Commission Report] {title}")
            n_ws += len(cs)
            print(f"  {title:52s} {len(cs):4d} chunks")
        print(f"  -> {n_ws} chunks from the Commission Report")

    # Provenance gates the detector. A chunk from a core 9/11 article may be
    # UPGRADED to forget by a strong hit, but may never be CLEARED into retain:
    # an audit of the first build found paragraphs describing the collapse
    # mechanism verbatim ("the entire building above fell onto the first intact
    # floor beneath impact") landing in retain purely because they said
    # "building" rather than "World Trade Center". Context is evidence the
    # detector cannot see, so core-article chunks with no strong hit are dropped.
    for c in all_chunks:
        if c.label == "general":
            # Safety net: a general article that somehow mentions 9/11 is dropped.
            if trace_hits(c.text):
                c.label = "dropped"
                dropped.append(c)
            else:
                c.hits = []
                retain.append(c)
            continue

        s = strong_hits(c.text)
        if s:
            c.hits = s
            c.label = "forget"
            forget.append(c)
        elif c.article in core:
            c.hits = trace_hits(c.text)
            c.label = "dropped_core_provenance"
            dropped.append(c)
        elif trace_hits(c.text):
            c.hits = trace_hits(c.text)
            c.label = "dropped"
            dropped.append(c)
        else:
            c.label = "retain"
            retain.append(c)

    def dump(name: str, chunks: list[Chunk]) -> str:
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({
                    "text": c.text,
                    "article": c.article,
                    "section": c.section,
                    "label": c.label,
                    "split": c.split,
                    "group": c.group,
                    "detector_hits": c.hits,
                }, ensure_ascii=False) + "\n")
        return path

    leaked = [c for c in retain if c.article in core]
    assert not leaked, f"provenance gate failed: {len(leaked)} core chunks in retain"

    # Redirects mean the same article can arrive under several requested titles
    # ("9/11 conspiracy theories", "9/11 conspiracies", "9-11 domestic
    # conspiracy theory" are one page), which duplicated 219 forget chunks and
    # would silently triple the weight of that content in the objective.
    import hashlib

    def dedupe(chunks: list[Chunk]) -> list[Chunk]:
        seen: set[str] = set()
        out_: list[Chunk] = []
        for c in chunks:
            h = hashlib.sha256(c.text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out_.append(c)
        return out_

    n_before = (len(forget), len(retain))
    forget = dedupe(forget)
    retain = dedupe(retain)
    print(f"\n  deduped: forget {n_before[0]}->{len(forget)}, "
          f"retain {n_before[1]}->{len(retain)}")

    p_f = dump("forget.jsonl", forget)
    p_r = dump("retain.jsonl", retain)
    p_d = dump("dropped.jsonl", dropped)

    total_chars = lambda cs: sum(len(c.text) for c in cs)  # noqa: E731
    print("\n" + "=" * 68)
    print("CORPUS SUMMARY")
    print("=" * 68)
    f_tr = [c for c in forget if c.split == "train"]
    f_ho = [c for c in forget if c.split == "holdout"]
    print(f"  forget : {len(forget):5d} chunks  {total_chars(forget):9,d} chars  -> {p_f}")
    print(f"           train={len(f_tr)}  holdout={len(f_ho)} "
          f"({len({c.article for c in f_ho})} reserved articles, never trained on)")
    print(f"  retain : {len(retain):5d} chunks  {total_chars(retain):9,d} chars  -> {p_r}")
    print(f"  dropped: {len(dropped):5d} chunks  {total_chars(dropped):9,d} chars  -> {p_d}")
    n_prov = sum(1 for c in dropped if c.label == "dropped_core_provenance")
    print(f"  (of dropped: {n_prov} by core-article provenance, "
          f"{len(dropped) - n_prov} by ambiguous 9/11 trace)")

    from collections import Counter
    print("\n  held-out forget articles (never trained on):")
    for a in sorted({c.article for c in f_ho}):
        print(f"    {a}")

    print("\n  forget chunks by source article (top 12):")
    for a, n in Counter(c.article for c in forget).most_common(12):
        print(f"    {a:55s} {n:4d}")
    print("\n  retain chunks pulled OUT of 9/11-adjacent articles (top 12):")
    adj = [c for c in retain if c.article in set(RETAIN_SOURCES)]
    for a, n in Counter(c.article for c in adj).most_common(12):
        print(f"    {a:55s} {n:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
