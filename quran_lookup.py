#!/usr/bin/env python3
"""quran_lookup.py — resolve canonical Qurʾānic text by surah:ayah.

Peer to hadith_lookup, for Qurʾānic block quotes. Uses fawazahmed0/quran-api
(CDN, no auth), Uthmani vocalized edition. A verse reference is definitive by
construction, so a successful keyed fetch verified by text-containment is
trusted. Reuses the text-normalisation utilities from hadith_lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hadith_lookup import _fetch_json, containment

CDN = "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1"
EDITION = "ara-quranuthmanihaf"  # Uthmani script, full tashkeel

# A verse reference (surah:ayah) is itself authoritative, so the text match is
# only a sanity guard against a mis-cited verse — not a strict verifier. Uthmani
# vs standard orthography diverges enough (dagger alef, etc.) that a strict 0.90
# would reject correct verses, so we use a lenient guard here.
QURAN_THRESHOLD = 0.55


@dataclass
class QuranResolution:
    matched: bool
    text: Optional[str] = None
    surah: Optional[int] = None
    ayah_start: Optional[int] = None
    ayah_end: Optional[int] = None
    score: float = 0.0
    method: str = ""


def fetch_ayah(surah: int, ayah: int) -> Optional[str]:
    """Return the Uthmani text of one ayah, or None on miss."""
    url = f"{CDN}/editions/{EDITION}/{surah}/{ayah}.json"
    try:
        return _fetch_json(url)["text"]
    except Exception:
        return None


def fetch_range(surah: int, ayah_start: int, ayah_end: int) -> Optional[str]:
    """Return the joined Uthmani text for an inclusive ayah range."""
    parts = []
    for a in range(ayah_start, ayah_end + 1):
        t = fetch_ayah(surah, a)
        if t is None:
            return None
        parts.append(t)
    return " ".join(parts) if parts else None


def resolve(
    surah: int,
    ayah_start: int,
    ayah_end: Optional[int],
    query_arabic: str,
    threshold: float = QURAN_THRESHOLD,
) -> QuranResolution:
    """Resolve a Qurʾānic block to canonical text, verified by containment."""
    if not surah or not ayah_start:
        return QuranResolution(False, method="flagged:no-verse-reference")
    ayah_end = ayah_end or ayah_start

    text = fetch_range(surah, ayah_start, ayah_end)
    if text is None:
        return QuranResolution(
            False, surah=surah, ayah_start=ayah_start, ayah_end=ayah_end,
            method="flagged:verse-fetch-failed",
        )

    score = containment(query_arabic, text) if query_arabic else 1.0
    if score >= threshold:
        # The citation defines exact ayah boundaries, so return the full fetched
        # range verbatim rather than truncating by an imperfect text alignment.
        return QuranResolution(
            True, text, surah, ayah_start, ayah_end, score, "quran-keyed"
        )

    return QuranResolution(
        False, surah=surah, ayah_start=ayah_start, ayah_end=ayah_end,
        score=score, method="flagged:verse-text-mismatch",
    )


if __name__ == "__main__":
    # Unit test: partial Āyat al-Kursī reading -> resolves to canonical 2:255.
    query = "الله لا إله إلا هو الحي القيوم لا تأخذه سنة ولا نوم"
    r = resolve(2, 255, 255, query)
    print(f"matched: {r.matched}  method: {r.method}  score: {r.score:.2f}")
    print(f"surah:ayah = {r.surah}:{r.ayah_start}")
    print(f"canonical: {r.text[:120] if r.text else None}...")
