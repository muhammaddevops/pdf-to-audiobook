#!/usr/bin/env python3
"""hadith_lookup.py — resolve canonical hadith Arabic from the open dataset.

Given a citation (collection + number) and GPT-4o's best-effort reading of the
Arabic block, fetch the *canonical* text from fawazahmed0/hadith-api (CDN, no
auth) and verify it is the right hadith by text-containment match. GPT's reading
is used only as a search key; the returned text is the trusted canonical string.

Strategy per block:
  1. Keyed lookup by number, verify containment >= threshold  -> trust it.
  2. If that fails (numbering mismatch), search the whole collection by text.
  3. If nothing clears the threshold (or the collection isn't in the dataset,
     e.g. Musnad Aḥmad), return None so the caller can flag it for review.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

CDN = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1"
DEFAULT_THRESHOLD = 0.90

# Map the transliterated collection names printed in books to dataset keys.
# Only collections present in the dataset are resolvable; others (e.g. ahmad)
# are intentionally absent and will fall through to a flagged result.
COLLECTION_ALIASES: dict[str, str] = {
    "bukhari": "bukhari",
    "sahih al-bukhari": "bukhari",
    "muslim": "muslim",
    "sahih muslim": "muslim",
    "abudawud": "abudawud",
    "abu dawud": "abudawud",
    "sunan abi dawud": "abudawud",
    "sunan abu dawud": "abudawud",
    "tirmidhi": "tirmidhi",
    "al-tirmidhi": "tirmidhi",
    "jami al-tirmidhi": "tirmidhi",
    "sunan al-tirmidhi": "tirmidhi",
    "jami at-tirmidhi": "tirmidhi",
    "nasai": "nasai",
    "al-nasai": "nasai",
    "sunan al-nasai": "nasai",
    "sunan an-nasai": "nasai",
    "ibnmajah": "ibnmajah",
    "ibn majah": "ibnmajah",
    "sunan ibn majah": "ibnmajah",
    "malik": "malik",
    "muwatta": "malik",
    "muwatta malik": "malik",
    "nawawi": "nawawi",
    "qudsi": "qudsi",
    "dehlawi": "dehlawi",
}


@dataclass
class Resolution:
    """Outcome of resolving one Arabic block against the canonical dataset."""

    matched: bool
    text: Optional[str] = None      # canonical Arabic, if matched
    collection: Optional[str] = None
    number: Optional[str] = None
    score: float = 0.0
    method: str = ""                # "keyed", "search", or "flagged:<reason>"


# Fold letter variants to a common skeleton so Uthmani (ٱ wasla, ٰ dagger alef,
# Quranic annotation marks) and standard imlāʾī orthography compare equal.
_LETTER_MAP = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٰ": "ا",  # alef forms + dagger
        "ى": "ا",                                            # alef maqsura
        "ة": "ه", "ؤ": "و", "ئ": "ي", "ء": "",                    # hamza/ta-marbuta
        "ـ": "",                                             # tatweel
    }
)


def normalize(s: str) -> str:
    """Reduce to a bare-letter skeleton for cross-orthography matching.

    Folds every alef/hamza variant and dagger alef to plain alef, drops tatweel,
    then removes all Unicode combining marks (harakat, small-high signs, Qurʾanic
    annotation). This lets Uthmani-script Qurʾan text match standard-script
    transcriptions and strips all tashkeel from hadith text alike.
    """
    s = s.translate(_LETTER_MAP)  # fold letter variants (incl. dagger alef -> alef)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # drop marks
    s = re.sub(r"[^ء-ي\s]", " ", s)   # keep base Arabic letters only
    return re.sub(r"\s+", " ", s).strip()

def containment(query: str, doc: str) -> float:
    """Fraction of the query's distinct words that appear in the document.

    Containment (not full similarity) because the page prints only the matn
    quote while the canonical entry also carries the isnād — the query is a
    substring of the document, so we score how much of the query is present.
    """
    q = set(normalize(query).split())
    d = set(normalize(doc).split())
    return len(q & d) / len(q) if q else 0.0


def _tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Word tokens as (normalized_form, start_offset, end_offset) in `text`."""
    out = []
    for m in re.finditer(r"\S+", text):
        nw = normalize(m.group(0))
        if nw:  # skip pure-punctuation tokens
            out.append((nw, m.start(), m.end()))
    return out


def extract_passage(canonical: str, query: str) -> str:
    """Return the span of `canonical` that covers `query`, in original spelling.

    The page prints only the matn quote, but the dataset entry also carries the
    isnād and grading commentary. We align the query's words to the canonical
    entry and return just the covered span (with its original tashkeel), so the
    output matches the scope of what's on the page. Falls back to the full text
    if no alignment is found.
    """
    spans = _tokens_with_spans(canonical)
    if not spans:
        return canonical
    canon_norm = [t[0] for t in spans]
    q_norm = normalize(query).split()
    sm = SequenceMatcher(a=canon_norm, b=q_norm, autojunk=False)
    matched = [b.a + k for b in sm.get_matching_blocks() for k in range(b.size)]
    if not matched:
        return canonical
    lo, hi = min(matched), max(matched)
    passage = canonical[spans[lo][1] : spans[hi][2]]
    return passage.strip(" \t\n\"“”‏‎.،؟!")


def resolve_collection_key(name: str) -> Optional[str]:
    """Map a printed collection name to a dataset key, or None if unsupported."""
    # Fold transliteration diacritics (ī→i, ḥ→h, ā→a, ...) before matching.
    folded = unicodedata.normalize("NFKD", name.strip().lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    key = re.sub(r"[^a-z\s-]", "", folded).strip()
    return COLLECTION_ALIASES.get(key)


def _fetch_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def fetch_by_number(collection_key: str, number: str) -> Optional[str]:
    """Return the canonical Arabic text for a hadith number, or None on miss."""
    url = f"{CDN}/editions/ara-{collection_key}/{number}.json"
    try:
        data = _fetch_json(url)
        return data["hadiths"][0]["text"]
    except Exception:
        return None


def _fetch_collection(collection_key: str) -> list[dict]:
    """Fetch every hadith in a collection (one file). Empty list on failure."""
    url = f"{CDN}/editions/ara-{collection_key}.json"
    try:
        return _fetch_json(url).get("hadiths", [])
    except Exception:
        return []


def resolve(
    collection_name: str,
    number: str,
    query_arabic: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> Resolution:
    """Resolve one Arabic block to canonical text with a verified match."""
    key = resolve_collection_key(collection_name)
    if key is None:
        return Resolution(
            matched=False,
            collection=collection_name,
            number=number,
            method="flagged:collection-not-in-dataset",
        )

    # 1) Trust the citation number, but verify by text.
    text = fetch_by_number(key, number)
    if text is not None:
        score = containment(query_arabic, text)
        if score >= threshold:
            passage = extract_passage(text, query_arabic)
            return Resolution(True, passage, key, number, score, "keyed")

    # 2) Numbering mismatch (or bad read): search the whole collection by text.
    best_text, best_num, best_score = None, None, 0.0
    for h in _fetch_collection(key):
        score = containment(query_arabic, h.get("text", ""))
        if score > best_score:
            best_text, best_num, best_score = h["text"], h.get("hadithnumber"), score
    if best_text is not None and best_score >= threshold:
        passage = extract_passage(best_text, query_arabic)
        return Resolution(True, passage, key, str(best_num), best_score, "search")

    # 3) Nothing trustworthy — flag for manual review / OCR fallback.
    return Resolution(
        matched=False,
        collection=key,
        number=number,
        score=best_score,
        method="flagged:no-match-above-threshold",
    )


if __name__ == "__main__":
    # End-to-end demo on page 1, hadith 1 (as printed: Sunan al-Tirmidhī 478).
    page_reading = "إنها ساعة تفتح فيها أبواب السماء وأحب أن يصعد لي فيها عمل صالح"
    print("Demo: resolving page-1 hadith 1 (cited 'Sunan al-Tirmidhī 478')\n")

    r = resolve("Sunan al-Tirmidhī", "478", page_reading)
    print(f"  matched : {r.matched}")
    print(f"  method  : {r.method}")
    print(f"  score   : {r.score:.2f}")
    print(f"  canon   : {r.text[:160] if r.text else None}...")

    print("\nDemo: Musnad Aḥmad 23565 (not in dataset — should flag)\n")
    r2 = resolve("Musnad Aḥmad", "23565", "إن أبواب السماء تفتح إذا زالت الشمس")
    print(f"  matched : {r2.matched}")
    print(f"  method  : {r2.method}")
