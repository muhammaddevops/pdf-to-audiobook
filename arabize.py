#!/usr/bin/env python3
"""arabize.py — convert Romanized Arabic terms in the script to Arabic script.

The narration voice mispronounces Latin-transliterated Arabic terms
(al-Tirmidhī, Ṣaḥīḥ, Abū Ayyūb, …). Rather than fold them to plain English (the
old narrate.py behaviour), we convert every Arabic-origin term to Arabic *script*
so the multilingual voice reads it natively, code-switching inside the sentence.

Two-stage, cheapest-first:

  1. DETERMINISTIC MAP (translit_map.py) — expand honorific glyphs to Arabic words
     and convert the common/known terms to bare Arabic script. Free, consistent,
     no API call. The map is a general common-Arabic-word list, not book-specific.
  2. LLM LONG TAIL (GPT-4o) — only if a page still has Latin-transliterated Arabic
     terms the map didn't cover (rare proper nouns). Pages fully handled by the map
     skip the call entirely.

Per the design decision, converted terms are left BARE — no harakat — and the TTS
voice supplies the vowels; injecting tashkeel only adds pronunciation-error risk.
The Arabic block quotes already printed on the page keep their vocalization.

  baseerOutput/*.txt → map (+ GPT-4o for the tail) → arabizedOutput/*.txt → narrate.py

Environment (.env):
  OPENAI_API_KEY   (required only for pages that hit the LLM long tail)

Usage:
  python arabize.py                    # converts ./baseerOutput → ./arabizedOutput
  python arabize.py path/to/folder     # convert a different source folder
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import translit_map

MODEL = "gpt-4o"
SRC_DIR = Path("baseerOutput")
OUT_DIR = Path("arabizedOutput")

LLM_PROMPT = (
    "You are an expert bilingual (Arabic/English) editor preparing a page for a "
    "text-to-speech engine. Most Arabic-origin terms have already been converted "
    "to Arabic script; a few Latin-transliterated ones remain. Convert ONLY those "
    "remaining Romanized Arabic-origin terms (names, place names, scholar names, "
    "citation titles, grading and technical terms) into Arabic script, following "
    "these rules EXACTLY:\n\n"
    "1. Do NOT add any tashkeel/harakat (vowel marks). Output BARE, unvocalized "
    "Arabic letters only — the TTS voice supplies the vowels.\n"
    "2. Leave every Arabic word ALREADY on the page EXACTLY as-is, character for "
    "character, including its existing harakat. Do NOT revocalize or edit it.\n"
    "3. Leave genuinely English words in English.\n"
    "4. Leave digits, markdown, parentheses, and punctuation untouched, and "
    "preserve the line and paragraph structure EXACTLY.\n\n"
    'Return ONLY a JSON object {"converted": "<the full rewritten page text>"}.'
)


def natural_key(path: Path) -> list:
    """Sort key so page2 < page10 (numeric chunks compared as ints)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]


def llm_convert(client, text: str, name: str) -> Optional[str]:
    """Convert the residual Romanized terms via GPT-4o. Returns text or None."""
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": f"{LLM_PROMPT}\n\n---PAGE---\n{text}"}],
        )
        import json
        data = json.loads(resp.choices[0].message.content or "{}")
        out = (data.get("converted") or "").strip()
        if not out:
            print(f"  ✗ {name}  model returned empty conversion", file=sys.stderr)
            return None
        print(f"    ↳ {name}  GPT-4o long-tail pass  {time.perf_counter() - start:.1f}s")
        return out
    except (APITimeoutError, APIConnectionError) as exc:
        print(f"  ✗ {name}  network/timeout: {exc}", file=sys.stderr)
    except RateLimitError as exc:
        print(f"  ✗ {name}  rate limited: {exc}", file=sys.stderr)
    except APIError as exc:
        print(f"  ✗ {name}  API error: {exc}", file=sys.stderr)
    except Exception as exc:  # final safety net (incl. bad JSON)
        print(f"  ✗ {name}  unexpected: {exc}", file=sys.stderr)
    return None


def main(argv: list[str]) -> int:
    load_dotenv()

    folder = Path(argv[1]) if len(argv) > 1 else SRC_DIR
    files = sorted((p for p in folder.glob("*.txt") if p.stat().st_size > 0), key=natural_key)
    if not files:
        print(f"[fatal] No non-empty .txt files in {folder}/", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)

    # Lazily build the LLM client only if a page actually needs the long tail, so
    # a map-only run needs no ANTHROPIC_API_KEY.
    client = None

    print(f"Arabizing {len(files)} page(s) from {folder}/ → {OUT_DIR}/\n")
    failures = 0
    map_only = 0
    llm_used = 0
    altered = 0
    for f in files:
        raw = f.read_text(encoding="utf-8")
        # Stage 1: deterministic — honorific glyphs → Arabic words, known terms → script.
        text = translit_map.expand_honorifics(raw)
        text, hits = translit_map.apply_map(text)

        # Stage 2: only if Latin-transliterated Arabic terms remain.
        if translit_map.has_translit_residue(text):
            if client is None:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    print(f"  ✗ {f.name}  needs the LLM long tail but OPENAI_API_KEY "
                          f"is unset — writing map-only output with residue.", file=sys.stderr)
                    failures += 1
                    (OUT_DIR / f.name).write_text(text, encoding="utf-8")
                    continue
                client = OpenAI(api_key=api_key, timeout=120.0)
            converted = llm_convert(client, text, f.name)
            if converted is None:
                failures += 1
                (OUT_DIR / f.name).write_text(text, encoding="utf-8")  # save map output anyway
                continue
            text = converted
            llm_used += 1
        else:
            map_only += 1

        # Safety: every pre-existing vocalized Arabic block line must survive verbatim.
        missing = [ln for ln in translit_map.arabic_lines(raw) if ln not in text]
        if missing:
            altered += 1
            print(f"    ⚠ {f.name}: {len(missing)} existing Arabic block line(s) changed "
                  f"— review before narrating.", file=sys.stderr)

        (OUT_DIR / f.name).write_text(text, encoding="utf-8")
        print(f"  ✓ {f.name}  {sum(hits.values())} term(s) mapped")

    print(f"\nDone. {map_only} map-only, {llm_used} used GPT-4o, {failures} failed, "
          f"{altered} page(s) with altered Arabic blocks.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
