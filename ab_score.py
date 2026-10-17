#!/usr/bin/env python3
"""ab_score.py — A/B the mixed-language read across ElevenLabs models and score it.

Renders the arabized script on each model (multilingual v2 vs eleven_v3), then
transcribes each chunk back with Whisper and scores how well the transcription
matches the text we sent. The score is an *intelligibility* proxy: if the voice
pronounced the Arabic terms and English prose clearly, Whisper recovers the words
and the score is high; garbled code-switching drops it. It does NOT measure
prosody/naturalness — judge smoothness by ear from the saved WAVs.

  arabizedOutput/*.txt → [render on model] → Whisper → score vs source

Outputs (in audio/ab/):
  <model>/full.wav            full render per model (listen to this)
  <model>/chunk_NN.wav        per-chunk renders (granular listening)
  ab_report.json             per-chunk + aggregate scores
  ab_report.md               human-readable scoreboard

Environment (.env):
  ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID   (render)
  OPENAI_API_KEY                            (Whisper transcription)

Usage:
  python ab_score.py                                   # v2 vs v3 on arabizedOutput/
  python ab_score.py eleven_multilingual_v2 eleven_v3  # explicit model list
  python ab_score.py --dir path/to/textdir <models...>
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

import narrate
from hadith_lookup import _LETTER_MAP  # fold alef/hamza/ta-marbuta variants

OUT_ROOT = Path("audio/ab")
DEFAULT_MODELS = ["eleven_multilingual_v2", "eleven_v3"]
WHISPER_MODEL = "whisper-1"


def normalize(s: str) -> str:
    """Fold to a bare comparison form: strip diacritics, fold Arabic letter
    variants, lowercase, keep only letters/digits, collapse whitespace. Applies to
    mixed Arabic+English so Whisper's diacritic-free output compares fairly."""
    s = s.translate(_LETTER_MAP)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^0-9a-z؀-ۿ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def score(expected: str, hypothesis: str) -> float:
    """Token-level similarity in [0, 1] between sent text and transcription."""
    return SequenceMatcher(None, normalize(expected).split(),
                           normalize(hypothesis).split()).ratio()


def transcribe(client, wav_path: Path) -> str:
    """Whisper transcription of a WAV, or '' on failure."""
    try:
        with open(wav_path, "rb") as fh:
            resp = client.audio.transcriptions.create(model=WHISPER_MODEL, file=fh)
        return resp.text or ""
    except Exception as exc:  # network / API — score this chunk 0, keep going
        print(f"      ✗ transcription failed: {exc}", file=sys.stderr)
        return ""


def render_and_score(el_key: str, voice: str, oa_client, model: str,
                     chunks: list[str]) -> dict:
    """Render every chunk on `model`, transcribe, and score. Returns a summary."""
    model_dir = OUT_ROOT / model
    model_dir.mkdir(parents=True, exist_ok=True)
    book_pcm = bytearray()
    rows = []
    for i, piece in enumerate(chunks):
        prev = chunks[i - 1] if i > 0 else None
        nxt = chunks[i + 1] if i + 1 < len(chunks) else None
        pcm = narrate.synthesize(el_key, voice, model, piece, prev, nxt)
        if pcm is None:
            rows.append({"chunk": i, "score": 0.0, "note": "render failed"})
            continue
        wav = model_dir / f"chunk_{i:02d}.wav"
        narrate.write_wav(wav, pcm)
        if book_pcm:
            book_pcm += narrate.silence(narrate.PAUSE_BETWEEN_CHUNKS)
        book_pcm += pcm
        hyp = transcribe(oa_client, wav)
        s = score(piece, hyp)
        rows.append({"chunk": i, "score": round(s, 4), "chars": len(piece),
                     "transcript": hyp[:200]})
        print(f"    chunk {i+1}/{len(chunks)}  score {s:.3f}")
    narrate.write_wav(model_dir / "full.wav", bytes(book_pcm))
    scored = [r["score"] for r in rows if "score" in r]
    mean = sum(scored) / len(scored) if scored else 0.0
    return {"model": model, "mean_score": round(mean, 4),
            "chunks": len(chunks), "rows": rows}


def main(argv: list[str]) -> int:
    load_dotenv()
    el_key = os.environ.get("ELEVENLABS_API_KEY")
    voice = os.environ.get("ELEVENLABS_VOICE_ID")
    oa_key = os.environ.get("OPENAI_API_KEY")
    if not (el_key and voice and oa_key):
        print("[fatal] Need ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, OPENAI_API_KEY.",
              file=sys.stderr)
        return 2

    args = argv[1:]
    folder = narrate.SRC_DIR
    if args and args[0] == "--dir":
        folder = Path(args[1])
        args = args[2:]
    models = args or DEFAULT_MODELS

    files = sorted((p for p in folder.glob("*.txt") if p.stat().st_size > 0),
                   key=narrate.natural_key)
    if not files:
        print(f"[fatal] No non-empty .txt files in {folder}/ "
              f"(run arabize.py first).", file=sys.stderr)
        return 1

    from openai import OpenAI
    oa_client = OpenAI(api_key=oa_key, timeout=120.0)
    document = narrate.preprocess(narrate.build_document(files))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in models:
        chunks = narrate.chunk(document, narrate.max_chars_for(model))
        print(f"\n=== {model}  ({len(chunks)} chunk(s)) ===")
        summaries.append(render_and_score(el_key, voice, oa_client, model, chunks))

    (OUT_ROOT / "ab_report.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable scoreboard.
    lines = ["# ElevenLabs A/B — intelligibility scoreboard", "",
             "STT round-trip score (higher = clearer pronunciation of the mixed "
             "Arabic/English read). Judge smoothness/prosody by ear from "
             "`audio/ab/<model>/full.wav`.", "",
             "| Model | Chunks | Mean score |", "|---|---|---|"]
    for s in sorted(summaries, key=lambda x: -x["mean_score"]):
        lines.append(f"| `{s['model']}` | {s['chunks']} | **{s['mean_score']:.3f}** |")
    winner = max(summaries, key=lambda x: x["mean_score"]) if summaries else None
    if winner:
        lines += ["", f"Top by intelligibility: **`{winner['model']}`** "
                  f"({winner['mean_score']:.3f}). Confirm by ear before committing."]
    (OUT_ROOT / "ab_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines[4:]))
    print(f"\n→ audio/ab/  (per-model full.wav + ab_report.md/json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
