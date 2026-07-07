#!/usr/bin/env python3
"""transcribe.py — book pages → TTS-ready bilingual script (Task 1).

Pipeline per page image in ./bookImages:

  1. GPT-4o editorial pass (JSON mode): produces the clean English + converted
     transliteration + expanded honorifics, with a ⟦BLOCK_n⟧ placeholder wherever
     a native Arabic quotation sat, plus a manifest of those blocks (GPT's
     best-effort Arabic + the citation it belongs to).
  2. Canonical resolution: each block is matched to the exact hadith in the open
     dataset (fawazahmed0/hadith-api) via keyed lookup + text-containment verify.
     GPT's Arabic is only a search key; the output is the trusted canonical text.
  3. Baseer OCR fallback: blocks with no canonical source (e.g. Musnad Aḥmad) are
     recovered by a specialist Arabic OCR pass — if credentials are configured.
  4. Stitch: resolved Arabic is substituted back into the placeholders.

Outputs (in ./output):
  page_XX.txt              final TTS-ready script
  page_XX.report.json      per-block provenance (method, score, source)
  full_transcript.txt      all pages concatenated in reading order

Usage:
  python transcribe.py                 # processes ./bookImages
  python transcribe.py path/to/folder
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

import hadith_lookup
import quran_lookup
from baseer_ocr import extract_arabic_blocks as baseer_extract
from opus_ocr import extract_arabic_blocks as opus_extract

# OCR fallback providers, tried in order for blocks with no canonical source.
# Each takes an image Path and returns list[str] | None. Both are best-effort
# (general/specialist VLMs), so their output is marked unverified in the report.
OCR_FALLBACKS = [("baseer", baseer_extract), ("opus", opus_extract)]

MODEL = "gpt-4o"
IMAGE_DIR = Path("bookImages")
OUTPUT_DIR = Path("output")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PLACEHOLDER_RE = re.compile(r"⟦BLOCK_(\d+)⟧")

TRANSCRIPTION_PROMPT = (
    "You are an expert bilingual editor, typesetter, and linguistic audio-script "
    "engineer. Examine the attached page image and produce a clean, read-ready "
    "script optimized for a text-to-speech engine. Follow these rules strictly.\n\n"
    "1. LANGUAGE SWAPPING FOR ACCENTS: Convert EVERY Arabic-origin Romanized term "
    "into its correct native Arabic script, with no exceptions, so the audio "
    "engine pronounces it in an Arabic voice. This includes personal names ('Abū "
    "Ayyūb' → 'أَبُو أَيُّوبَ', 'ʿAbdullāh ibn al-Sāʿib' → 'عَبْدُ اللَّهِ بْنُ "
    "السَّائِبِ'), source-citation titles ('Sunan al-Tirmidhī' → 'سُنَنُ "
    "التِّرْمِذِيِّ', 'Musnad Aḥmad' → 'مُسْنَدُ أَحْمَدَ'), grading and technical "
    "terms ('ḥadīth' → 'حَدِيثٌ', 'Ṣaḥīḥ' → 'صَحِيحٌ'), and scholar names "
    "('al-Arnāʾūṭ' → 'الْأَرْنَؤُوطُ'). Do NOT leave such terms in Latin. Keep "
    "numerals as Western digits. Only genuinely English words stay in English.\n"
    "2. VOCALIZATION (HIGH-CONFIDENCE ONLY): Add tashkeel/harakat to the Arabic you "
    "generate in rules 1 and 3, but only marks you are 100% certain of; leave a "
    "letter bare rather than guessing. Fewer trustworthy marks beat full risky "
    "vocalization.\n"
    "3. GLYPH EXPANSION: Expand inline honorific glyphs into vocalized Arabic words "
    "(ﷺ → 'صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ'; ؓ → 'رَضِيَ اللَّهُ عَنْهُ').\n"
    "4. NATIVE ARABIC QUOTES → PLACEHOLDER: Whenever you encounter a standalone "
    "Arabic quotation already printed on the page (hadith matn, Qurʾānic passage, "
    "vocalized block quote), do NOT transcribe it inline. Instead replace the whole "
    "block with a placeholder token '⟦BLOCK_n⟧' (n = 1, 2, 3... in reading order) "
    "and record it in the 'blocks' array (see OUTPUT FORMAT). Keep the surrounding "
    "English prose flowing naturally around the placeholder.\n"
    "5. LAYOUT CLEANUP: Strip page numbers, running headers, footers, and footnote "
    "markers. Separate paragraphs with a double line break.\n\n"
    "OUTPUT FORMAT: Return ONLY a JSON object with two keys:\n"
    '  "script": the final read-ready text (English + converted Arabic terms + '
    "'⟦BLOCK_n⟧' placeholders where native Arabic quotes were).\n"
    '  "blocks": an array; one object per placeholder, in order, each with:\n'
    '     "id": the integer n matching its ⟦BLOCK_n⟧ placeholder,\n'
    '     "type": "quran" if the block is a Qurʾānic verse/passage, "hadith" if '
    "it is a hadith quotation, else \"other\",\n"
    '     "arabic_best_effort": your best transcription of that Arabic block '
    "(used only to locate the canonical text — accuracy helps but need not be "
    "perfect),\n"
    '     "collection": for a hadith, the collection as a plain transliterated '
    "English name if a citation is present (e.g. 'Sunan al-Tirmidhi', 'Musnad "
    "Ahmad'), else null,\n"
    '     "number": for a hadith, the hadith number from the citation as a '
    "string, else null,\n"
    '     "surah": for a Qurʾān verse, the surah number (integer) if identifiable '
    "from a citation or the text, else null,\n"
    '     "ayah_start": for a Qurʾān verse, the first ayah number (integer), else '
    "null,\n"
    '     "ayah_end": for a Qurʾān verse spanning multiple ayahs, the last ayah '
    "number (integer); for a single ayah use the same value as ayah_start; else "
    "null.\n"
    "If there are no native Arabic quotation blocks, return an empty 'blocks' array."
)


def natural_key(path: Path) -> list:
    """Sort key so page2 < page10 (numeric chunks compared as ints)."""
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", path.name)
    ]


def find_images(folder: Path) -> list[Path]:
    """Return image files in the folder, sorted in natural page order."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    images = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=natural_key)


def encode_image(image_path: Path) -> str:
    """Return an RFC-2397 data URL for the image."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None or not mime_type.startswith("image/"):
        fallback = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = fallback.get(image_path.suffix.lower(), "image/png")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def transcribe_page(client: OpenAI, image_path: Path) -> Optional[dict]:
    """GPT-4o editorial pass. Returns {'script': str, 'blocks': list} or None."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(image_path),
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        elapsed = time.perf_counter() - start
        data = json.loads(response.choices[0].message.content or "{}")
        script = (data.get("script") or "").strip()
        blocks = data.get("blocks") or []
        print(
            f"  ✓ {image_path.name}  {elapsed:.1f}s  "
            f"({len(script)} chars, {len(blocks)} Arabic block(s))"
        )
        return {"script": script, "blocks": blocks}
    except (APITimeoutError, APIConnectionError) as exc:
        print(f"  ✗ {image_path.name}  network/timeout: {exc}", file=sys.stderr)
    except RateLimitError as exc:
        print(f"  ✗ {image_path.name}  rate limited: {exc}", file=sys.stderr)
    except APIError as exc:
        code = getattr(exc, "status_code", None)
        print(f"  ✗ {image_path.name}  API error ({code}): {exc}", file=sys.stderr)
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"  ✗ {image_path.name}  bad JSON from model: {exc}", file=sys.stderr)
    except Exception as exc:  # final safety net
        print(f"  ✗ {image_path.name}  unexpected: {exc}", file=sys.stderr)
    return None


def resolve_blocks(blocks: list[dict], image_path: Path) -> dict[int, dict]:
    """Resolve each Arabic block to canonical text (or Baseer fallback / flag).

    Returns {block_id: {'text': str|None, 'method': str, 'score': float}}.
    """
    resolved: dict[int, dict] = {}
    unresolved: list[dict] = []

    # Pass 1: canonical dataset lookup + text verification, routed by block type.
    for b in blocks:
        bid = b.get("id")
        query = b.get("arabic_best_effort") or ""
        btype = (b.get("type") or "").lower()

        if btype == "quran" and b.get("surah") and b.get("ayah_start"):
            qr = quran_lookup.resolve(
                int(b["surah"]),
                int(b["ayah_start"]),
                int(b["ayah_end"]) if b.get("ayah_end") else None,
                query,
            )
            if qr.matched:
                print(
                    f"    block {bid}: canonical Qurʾān {qr.surah}:{qr.ayah_start}"
                    f"-{qr.ayah_end} (score {qr.score:.2f})"
                )
                resolved[bid] = {
                    "text": qr.text, "method": qr.method, "score": qr.score
                }
                continue
            print(f"    block {bid}: no Qurʾān match ({qr.method}) → fallback")
            unresolved.append(b)
            continue

        collection = b.get("collection")
        number = b.get("number")
        if collection and number:
            r = hadith_lookup.resolve(collection, str(number), query)
            if r.matched:
                print(
                    f"    block {bid}: canonical via {r.method} "
                    f"(score {r.score:.2f}, {r.collection} {r.number})"
                )
                resolved[bid] = {"text": r.text, "method": r.method, "score": r.score}
                continue
            print(f"    block {bid}: no canonical match ({r.method}) → fallback")
        else:
            print(f"    block {bid}: no citation on page → fallback")
        unresolved.append(b)

    # Pass 2: OCR fallbacks (Baseer specialist, then Opus 4.8) for blocks with no
    # canonical source. Each provider reads the whole page; we align its returned
    # blocks to the right placeholder by text containment against GPT's reading.
    if unresolved:
        for provider, extract in OCR_FALLBACKS:
            if not unresolved:
                break
            ocr_blocks = extract(image_path)
            if not ocr_blocks:  # None (unavailable) or empty — try next provider
                continue
            still: list[dict] = []
            for b in unresolved:
                bid = b.get("id")
                query = b.get("arabic_best_effort") or ""
                best_text, best_score = None, 0.0
                for candidate in ocr_blocks:
                    score = hadith_lookup.containment(query, candidate)
                    if score > best_score:
                        best_text, best_score = candidate, score
                if best_text is not None and best_score >= hadith_lookup.DEFAULT_THRESHOLD:
                    print(f"    block {bid}: recovered via {provider} "
                          f"(score {best_score:.2f}, unverified)")
                    resolved[bid] = {
                        "text": best_text, "method": provider, "score": best_score
                    }
                else:
                    still.append(b)
            unresolved = still

        for b in unresolved:
            bid = b.get("id")
            print(f"    block {bid}: UNRESOLVED — omitted + flagged")
            resolved[bid] = {"text": None, "method": "flagged", "score": 0.0}
    return resolved


def stitch(script: str, resolved: dict[int, dict]) -> str:
    """Replace ⟦BLOCK_n⟧ placeholders with resolved Arabic (or drop if flagged)."""

    def sub(match: re.Match) -> str:
        bid = int(match.group(1))
        entry = resolved.get(bid)
        return entry["text"] if entry and entry.get("text") else ""

    out = PLACEHOLDER_RE.sub(sub, script)
    # Tidy whitespace left by dropped placeholders.
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def main(argv: list[str]) -> int:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[fatal] OPENAI_API_KEY not set (check your .env).", file=sys.stderr)
        return 2

    folder = Path(argv[1]) if len(argv) > 1 else IMAGE_DIR
    try:
        images = find_images(folder)
    except FileNotFoundError as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2

    if not images:
        print(
            f"[fatal] No image files found in {folder}/ "
            f"(looked for: {', '.join(sorted(IMAGE_EXTENSIONS))}). "
            "Drop the book page images in there and re-run.",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)
    client = OpenAI(api_key=api_key, timeout=120.0)

    print(f"Processing {len(images)} page(s) from {folder}/ with {MODEL}\n")
    pages: list[tuple[Path, str]] = []
    failures = 0
    total_flagged = 0
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}")
        result = transcribe_page(client, image_path)
        if result is None:
            failures += 1
            continue

        resolved = resolve_blocks(result["blocks"], image_path)
        final_text = stitch(result["script"], resolved)

        out_path = OUTPUT_DIR / f"page_{index:02d}.txt"
        out_path.write_text(final_text, encoding="utf-8")

        report = {
            "image": image_path.name,
            "blocks": [
                {
                    "id": b.get("id"),
                    "collection": b.get("collection"),
                    "number": b.get("number"),
                    **{k: resolved.get(b.get("id"), {}).get(k)
                       for k in ("method", "score")},
                }
                for b in result["blocks"]
            ],
        }
        (OUTPUT_DIR / f"page_{index:02d}.report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        flagged = [b["id"] for b in report["blocks"] if b["method"] == "flagged"]
        total_flagged += len(flagged)
        if flagged:
            print(f"  ⚠ page {index}: {len(flagged)} block(s) flagged: {flagged}")
        pages.append((out_path, final_text))

    if pages:
        combined = "\n\n".join(text for _, text in pages)
        full_path = OUTPUT_DIR / "full_transcript.txt"
        full_path.write_text(combined, encoding="utf-8")
        print(f"\nWrote {len(pages)} page file(s) + {full_path}")

    print(
        f"Done. {len(pages)} succeeded, {failures} failed, "
        f"{total_flagged} Arabic block(s) flagged for review."
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
