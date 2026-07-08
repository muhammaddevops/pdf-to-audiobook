#!/usr/bin/env python3
"""arabize.py — convert Romanized Arabic terms in the script to Arabic script.

The narration voice mispronounces Latin-transliterated Arabic terms
(al-Tirmidhī, Ṣaḥīḥ, Abū Ayyūb, …). Rather than fold them to plain English (the
old narrate.py behaviour), we convert every Arabic-origin term to Arabic *script*
so the multilingual voice reads it natively, code-switching inside the sentence.

An LLM does the conversion because the term set is open-ended (names, place
names, scholar names, citation titles, grading + technical terms) and a hand
dictionary (convert_translit.py) can't keep up. Per the design decision, the
converted terms are left BARE — no harakat — and the TTS voice supplies the
vowels; injecting tashkeel would only add pronunciation-error risk. The Arabic
block quotes already printed on the page keep their existing vocalization,
untouched.

  baseerOutput/*.txt → (GPT-4o, JSON) → arabizedOutput/*.txt → narrate.py

Outputs mirror the input filenames in ./arabizedOutput. A per-file check warns if
any pre-existing vocalized Arabic block was altered by the model.

Environment (.env):
  OPENAI_API_KEY   (required)

Usage:
  python arabize.py                    # converts ./baseerOutput → ./arabizedOutput
  python arabize.py path/to/folder     # convert a different source folder
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

MODEL = "gpt-4o"
SRC_DIR = Path("baseerOutput")
OUT_DIR = Path("arabizedOutput")

ARABIZE_PROMPT = (
    "You are an expert bilingual (Arabic/English) editor preparing a script for a "
    "text-to-speech engine. You will be given the raw text of one book page: "
    "English prose, Arabic-origin terms written in Latin transliteration, "
    "fully-vocalized Arabic quotation blocks, honorific glyphs, and citations. "
    "Rewrite the page by converting Arabic-origin terms into Arabic script, "
    "following these rules EXACTLY:\n\n"
    "1. CONVERT every Arabic-origin Romanized term into Arabic script: personal "
    "names, place names, scholar names, source-citation titles, hadith-grading "
    "terms, and technical religious terms. Examples: 'al-Tirmidhī' → 'الترمذي', "
    "'Sunan al-Tirmidhi' → 'سنن الترمذي', 'Ṣaḥīḥ' → 'صحيح', 'Abū Ayyūb' → "
    "'أبو أيوب', 'ḥadīth' → 'حديث', 'Ramadān' → 'رمضان', 'al-Albānī' → 'الألباني'.\n"
    "2. Do NOT add any tashkeel/harakat (vowel marks) to the Arabic you generate. "
    "Output BARE, unvocalized Arabic letters only — the TTS voice supplies the "
    "vowels. Adding harakat is a mistake.\n"
    "3. Leave any Arabic text ALREADY printed on the page (the vocalized quotation "
    "blocks) EXACTLY as-is — character for character, including all of its existing "
    "harakat. Do NOT revocalize, correct, reorder, add to, or drop anything inside "
    "those blocks.\n"
    "4. Leave genuinely English / non-Arabic words in English (e.g. 'The Messenger "
    "of Allah', 'reported', 'chapter', and proper nouns of non-Arabic origin).\n"
    "5. Leave honorific glyphs (ﷺ, ؓ, ﷻ, and the bracketed 'رضي الله عنه'), Western "
    "digits, markdown markers (**, *), parentheses, and punctuation untouched.\n"
    "6. Preserve the paragraph and line structure EXACTLY: same line breaks, same "
    "order, no summarizing, no translation of meaning, no added commentary.\n\n"
    'OUTPUT FORMAT: return ONLY a JSON object {"converted": "<the full rewritten '
    'page text>"}.'
)


def natural_key(path: Path) -> list:
    """Sort key so page2 < page10 (numeric chunks compared as ints)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]


def arabic_lines(text: str) -> list[str]:
    """Lines that are majority Arabic (the pre-existing block quotes)."""
    out = []
    for line in text.splitlines():
        letters = [c for c in line if c.strip()]
        arab = [c for c in line if "؀" <= c <= "ۿ"]
        if letters and len(arab) / len(letters) > 0.6:
            out.append(line.strip())
    return out


def arabize_page(client: OpenAI, text: str, name: str) -> Optional[str]:
    """Convert one page's Romanized terms to Arabic script. Returns text or None."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": f"{ARABIZE_PROMPT}\n\n---PAGE---\n{text}"}
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        converted = (data.get("converted") or "").strip()
        if not converted:
            print(f"  ✗ {name}  model returned empty conversion", file=sys.stderr)
            return None
        print(f"  ✓ {name}  {time.perf_counter() - start:.1f}s  ({len(converted)} chars)")
        return converted
    except (APITimeoutError, APIConnectionError) as exc:
        print(f"  ✗ {name}  network/timeout: {exc}", file=sys.stderr)
    except RateLimitError as exc:
        print(f"  ✗ {name}  rate limited: {exc}", file=sys.stderr)
    except APIError as exc:
        code = getattr(exc, "status_code", None)
        print(f"  ✗ {name}  API error ({code}): {exc}", file=sys.stderr)
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"  ✗ {name}  bad JSON from model: {exc}", file=sys.stderr)
    except Exception as exc:  # final safety net
        print(f"  ✗ {name}  unexpected: {exc}", file=sys.stderr)
    return None


def main(argv: list[str]) -> int:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[fatal] OPENAI_API_KEY not set (check your .env).", file=sys.stderr)
        return 2

    folder = Path(argv[1]) if len(argv) > 1 else SRC_DIR
    files = sorted((p for p in folder.glob("*.txt") if p.stat().st_size > 0), key=natural_key)
    if not files:
        print(f"[fatal] No non-empty .txt files in {folder}/", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    client = OpenAI(api_key=api_key, timeout=120.0)
    print(f"Arabizing {len(files)} page(s) from {folder}/ → {OUT_DIR}/ with {MODEL}\n")

    failures = 0
    altered = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        converted = arabize_page(client, text, f.name)
        if converted is None:
            failures += 1
            continue

        # Safety: every pre-existing vocalized Arabic block line must survive verbatim.
        missing = [ln for ln in arabic_lines(text) if ln not in converted]
        if missing:
            altered += 1
            print(f"    ⚠ {f.name}: {len(missing)} existing Arabic block line(s) "
                  f"changed by the model — review before narrating.", file=sys.stderr)

        (OUT_DIR / f.name).write_text(converted, encoding="utf-8")

    ok = len(files) - failures
    print(f"\nDone. {ok} converted, {failures} failed, {altered} page(s) with "
          f"altered Arabic blocks (flagged above).")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
