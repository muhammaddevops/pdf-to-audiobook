#!/usr/bin/env python3
"""narrate.py — bilingual text → smooth storyteller audio via ElevenLabs.

Single-voice pipeline: arabized script → normalize for TTS → one warm storyteller
voice reads the whole thing, code-switching inside the sentence — English prose in
English, and the inline Arabic-script terms (from arabize.py) recited natively
alongside the vocalized Arabic block quotes.

  arabizedOutput/*.txt → preprocess() → ElevenLabs (one call per chunk) → audio/

Run arabize.py first so Arabic-origin terms are already in Arabic script; the
voice then pronounces them natively instead of mangling the transliteration.
Set ELEVENLABS_MODEL=eleven_v3 to A/B the read against multilingual v2.

Normalization (deterministic, no LLM):
  * Honorific glyphs expanded to bare Arabic words (ﷺ → صلى الله عليه وسلم),
    recited natively, with the bracketing pause dropped (translit_map).
  * fold_latin() now only cleans stray transliteration diacritics arabize.py left
    behind (converted terms are already Arabic script, so it no longer touches them).
  * Markdown emphasis stripped; citations and headers kept and read verbatim.

Audio is assembled from raw PCM with the stdlib `wave` module (no ffmpeg/pydub):
one WAV per page plus a stitched full_book.wav.

Environment (.env):
  ELEVENLABS_API_KEY   (required)
  ELEVENLABS_VOICE_ID  (required) — the storyteller voice; set to Daniel
                       (onwK4e9ZLuTAKqWW03F9) for the voice we settled on.

Usage:
  python narrate.py                    # narrates ./baseerOutput
  python narrate.py path/to/folder
"""

from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

import translit_map

SRC_DIR = Path("arabizedOutput")   # arabize.py output: Arabic-script inline terms
OUT_DIR = Path("audio")
DEFAULT_MODEL = "eleven_multilingual_v2"
OUTPUT_FORMAT = "pcm_24000"        # 24 kHz, 16-bit mono PCM
SAMPLE_RATE = 24000
API_ROOT = "https://api.elevenlabs.io/v1/text-to-speech"

# Per-request character ceiling. eleven_v3 caps at 3000 chars/request (and drops
# SSML break tags); multilingual v2 comfortably takes more. Pick by model so the
# same text A/Bs cleanly across both. See docs/tts_narration_best_practices.md.
MAX_CHARS_BY_MODEL = {"eleven_v3": 2800}
MAX_CHARS_DEFAULT = 4500           # split a page above this at paragraph breaks


def max_chars_for(model: str) -> int:
    return MAX_CHARS_BY_MODEL.get(model, MAX_CHARS_DEFAULT)

# Smooth, clear storyteller feel: steady, high fidelity, a touch of warmth,
# slightly slower than default for a measured, narrated pace.
VOICE_SETTINGS = {
    "stability": 0.50,
    "similarity_boost": 0.80,
    "style": 0.15,
    "use_speaker_boost": True,
    "speed": 1.0,
}

PAUSE_BETWEEN_CHUNKS = 0.35        # small pause at chunk seams (paragraph breaks)

# --------------------------------------------------------------------------- #
# Pacing controls — multilingual v2 infers pauses from punctuation, so written
# text over-pauses when read aloud. These drop grammatical-but-not-prosodic
# breaks so Daniel flows. See docs/tts_narration_best_practices.md. Toggle to A/B.
# --------------------------------------------------------------------------- #
# Honorific inlining (drop the bracketing pause) now happens in translit_map's
# expand_honorifics, which also expands the glyphs to Arabic — so it's no longer a
# toggle here. THIN_CITATION_COMMAS remains an A/B knob.
# Off by default: by the pause table in the best-practices doc, an em dash and a
# comma are BOTH "short pause", so swapping them barely changes pacing — and the
# citation parens still pause on entry/exit regardless. It also alters citation
# text we chose to keep verbatim. Flip on to A/B if you want.
THIN_CITATION_COMMAS = False


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

# Modifier-letter ʿayn/hamza transliteration marks — dropped for the reader.
_AYN_HAMZA = {"ʿ", "ʾ", "ʼ", "ʹ", "ʻ"}

# Curated corrections for the few terms Baseer *mis*-transliterates — an explicit
# exception list, grown by hand as new mistakes are spotted (NOT a general rule).
CORRECTIONS = {
    "Suhr": "Zuhr",   # Baseer writes Ṣ for ظ; the Ẓuhr prayer
}


def fold_latin(text: str) -> str:
    """Fold Latin transliteration diacritics to plain ASCII for the English voice.

    'al-Tirmidhī' → 'al-Tirmidhi', 'Ṣaḥīḥ' → 'Sahih', 'Aḥmad' → 'Ahmad'. Arabic
    script and its harakat are untouched — those characters are not 'LATIN', so
    the fold skips them. Modifier ʿayn/hamza marks are dropped.
    """
    out = []
    for ch in text:
        if ch in _AYN_HAMZA:
            continue
        if unicodedata.name(ch, "").startswith("LATIN"):
            out.append("".join(c for c in unicodedata.normalize("NFKD", ch)
                               if not unicodedata.combining(c)))
        else:
            out.append(ch)
    folded = "".join(out)
    for wrong, right in CORRECTIONS.items():
        folded = re.sub(rf"\b{wrong}\b", right, folded)
    return folded


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")


def _under_1000(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (f"-{_ONES[o]}" if o else "")
    h, r = divmod(n, 100)
    return f"{_ONES[h]} hundred" + (f" {_under_1000(r)}" if r else "")


def number_to_words(n: int) -> str:
    """Cardinal English words for 0–999,999 (covers section + reference numbers)."""
    if n == 0:
        return "zero"
    parts = []
    for div, name in ((1000, "thousand"),):
        if n >= div:
            q, n = divmod(n, div)
            parts.append(f"{_under_1000(q)} {name}")
    if n:
        parts.append(_under_1000(n))
    return " ".join(parts)


def normalize_numbers(text: str) -> str:
    """Spell out ASCII-digit runs so ElevenLabs' own number engine never runs.

    Its auto-normalization is unreliable on bare section markers ('3.' → 'tigga')
    and large reference numbers ('2023' → 'two thousand unto twenty-three'). We
    only touch ASCII 0-9 (English citations/headings); Arabic-Indic digits, if any
    ever appear in the recitation blocks, are left untouched.
    """
    return re.sub(r"[0-9]+", lambda m: number_to_words(int(m.group())), text)


def smooth_pacing(text: str) -> str:
    """Drop grammatical-but-not-prosodic pauses so the narrator flows.

    multilingual v2 pauses on parentheses and commas; citations therefore chop
    the sentence. Honorific de-bracketing now happens in translit_map (which
    expands the glyphs to Arabic words and drops their parentheses); this handles
    the citation grade comma, which is conservative and toggle-guarded.
    """
    if THIN_CITATION_COMMAS:
        # "(Source 478, Sahih)" → "(Source 478 — Sahih)": an em dash separates
        # the grade without the full comma stop at the end of every quote.
        text = re.sub(
            r"\(([^()]*?),\s*((?:Sahih|Hasan|Da'?if|Mawdu'?)[^()]*)\)",
            r"(\1 — \2)",
            text,
        )
    return text


def preprocess(text: str) -> str:
    """Expand honorifics to Arabic words, strip markdown, and clean stray Latin
    transliteration diacritics for the narrator."""
    # Honorific glyphs → bare Arabic words (recited natively), brackets dropped.
    # Idempotent: arabize.py already applied this, so on arabized text it's a no-op.
    text = translit_map.expand_honorifics(text)
    # Baseer sometimes drops the honorific entirely, leaving empty parens (maybe
    # with an invisible RTL/LRM direction mark inside) — recover it as Arabic.
    text = re.sub(r"\([\s‎‏؜]*\)", "رضي الله عنه", text)
    # Strip markdown emphasis markers.
    text = re.sub(r"\*+", "", text)
    # Clean any stray transliteration diacritics arabize.py left on missed terms
    # (converted terms are already Arabic script, so this no longer folds names).
    text = fold_latin(text)
    # Spell numbers out so ElevenLabs' number normalizer never runs.
    text = normalize_numbers(text)
    # Drop grammatical-but-not-prosodic pauses (citation comma; toggle-guarded).
    text = smooth_pacing(text)
    # Normalise whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text: str, max_chars: int = MAX_CHARS_DEFAULT) -> list[str]:
    """Group paragraphs into request-sized chunks, splitting only at blank lines.

    Most pages fit in a single chunk (one API call → smoothest prosody); only
    very long pages are split, and always at paragraph boundaries.
    """
    chunks: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_document(files: list[Path]) -> str:
    """Merge page files into one continuous document.

    Page boundaries are print artifacts — a page frequently ends mid-sentence.
    Where a page's last paragraph does NOT end on terminal punctuation and the
    next page does NOT start a new numbered section, the split is a continuation:
    we join those paragraphs with a space so the sentence (or Arabic recitation)
    flows unbroken. Otherwise a paragraph break is kept. Chunking then happens on
    this merged text, so no cut ever lands on a mid-sentence page seam.
    """
    paras: list[str] = []
    for f in files:
        page_paras = [p.strip() for p in re.split(r"\n\s*\n", f.read_text(encoding="utf-8")) if p.strip()]
        if not page_paras:
            continue
        if paras:
            ends_hard = paras[-1].rstrip()[-1:] in ".!?)”\""
            new_section = bool(re.match(r"^\**\s*\d+\.", page_paras[0]))
            if not ends_hard and not new_section:
                paras[-1] = f"{paras[-1]} {page_paras[0]}"   # continuation → join
                paras.extend(page_paras[1:])
                continue
        paras.extend(page_paras)
    return "\n\n".join(paras)


# --------------------------------------------------------------------------- #
# ElevenLabs synthesis
# --------------------------------------------------------------------------- #

def synthesize(
    api_key: str,
    voice_id: str,
    model: str,
    text: str,
    prev_text: Optional[str],
    next_text: Optional[str],
    timeout: float = 180.0,
) -> Optional[bytes]:
    """Return raw PCM bytes for one chunk, or None on failure."""
    body = {"text": text, "model_id": model, "voice_settings": VOICE_SETTINGS}
    if prev_text:
        body["previous_text"] = prev_text
    if next_text:
        body["next_text"] = next_text
    try:
        resp = requests.post(
            f"{API_ROOT}/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            params={"output_format": OUTPUT_FORMAT},
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        detail = exc.response.text[:300] if exc.response is not None else ""
        print(f"    ✗ HTTP {code}: {detail}", file=sys.stderr)
    except requests.exceptions.RequestException as exc:
        print(f"    ✗ network error: {exc}", file=sys.stderr)
    return None


# --------------------------------------------------------------------------- #
# Audio assembly (stdlib wave, no ffmpeg)
# --------------------------------------------------------------------------- #

def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * SAMPLE_RATE)


def write_wav(path: Path, pcm: bytes) -> float:
    """Write mono 16-bit PCM to a WAV file; return duration in seconds."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return len(pcm) / 2 / SAMPLE_RATE


def natural_key(path: Path) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main(argv: list[str]) -> int:
    load_dotenv()
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    voice = os.environ.get("ELEVENLABS_VOICE_ID")
    if not api_key or not voice:
        print("[fatal] ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set in .env.",
              file=sys.stderr)
        return 2
    # A/B the read across models: `ELEVENLABS_MODEL=eleven_v3 python narrate.py`.
    model = os.environ.get("ELEVENLABS_MODEL", DEFAULT_MODEL)
    max_chars = max_chars_for(model)

    folder = Path(argv[1]) if len(argv) > 1 else SRC_DIR
    files = sorted((p for p in folder.glob("*.txt") if p.stat().st_size > 0), key=natural_key)
    if not files:
        print(f"[fatal] No non-empty .txt files in {folder}/", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)

    # Merge all pages into one continuous document (sentences flow across page
    # seams), then chunk by size at paragraph boundaries — so no cut lands
    # mid-sentence. previous_text/next_text carries prosody across chunk seams.
    chunks = chunk(preprocess(build_document(files)), max_chars)
    print(f"Narrating {len(files)} page(s) from {folder}/ as one continuous book "
          f"→ {len(chunks)} chunk(s)  ({model}, voice {voice[:8]}…)\n")

    book_pcm = bytearray()
    failures = 0
    for i, piece in enumerate(chunks):
        prev = chunks[i - 1] if i > 0 else None
        nxt = chunks[i + 1] if i + 1 < len(chunks) else None
        t0 = time.perf_counter()
        pcm = synthesize(api_key, voice, model, piece, prev, nxt)
        if pcm is None:
            failures += 1
            continue
        print(f"  ✓ chunk {i+1}/{len(chunks)}  {len(piece):>4} chars → "
              f"{len(pcm)/2/SAMPLE_RATE:5.1f}s  ({time.perf_counter()-t0:.1f}s)")
        if book_pcm:
            book_pcm += silence(PAUSE_BETWEEN_CHUNKS)
        book_pcm += pcm

    # Name the render after the model so v2/v3 A/B runs don't overwrite each other.
    book_path = OUT_DIR / f"full_book_{model}.wav"
    total = write_wav(book_path, bytes(book_pcm))
    print(f"\n→ {book_path}  ({total/60:.1f} min continuous)")
    print(f"Done. {failures} chunk(s) failed." if failures else "Done. All chunks OK.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
