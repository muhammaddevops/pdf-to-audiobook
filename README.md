# audiobook-render

Turn scanned pages of bilingual (Arabic / English) Islamic books into a smooth,
single-voice audiobook. Built for pages with heavy academic transliteration,
vocalized Arabic hadith/Qurʾān blocks, and inline honorific glyphs — the things
that trip up naive OCR + TTS.

## Pipeline (current, single-voice)

```
bookImages/*.png                     page scans
      │  Baseer OCR (specialist Arabic OCR, faithful on vocalized script)
      ▼
baseerOutput/*.txt                   English prose + Arabic blocks + romanized terms
      │  arabize.py:  convert Arabic-origin Romanized terms (al-Tirmidhī →
      │               الترمذي, Ṣaḥīḥ → صحيح) + honorific glyphs (ﷺ → صلى الله
      │               عليه وسلم) to BARE Arabic script — no harakat, the voice
      │               vowels them; existing vocalized Arabic blocks untouched.
      │               Deterministic map first (translit_map.py); Claude Opus only
      │               for pages with terms the map doesn't cover.
      ▼
arabizedOutput/*.txt                 English prose + Arabic-script inline terms + blocks
      │  narrate.py  (deterministic, no LLM):
      │    • build_document()  — merge pages into one continuous document so
      │                          sentences flow across page seams
      │    • preprocess()      — expand honorifics inline (ﷺ → "peace be upon him"),
      │                          spell out numbers, strip markdown, clean stray marks
      │    • chunk + request-stitch, one ElevenLabs call per chunk
      ▼
audio/full_book_<model>.wav          one continuous narration  →  .mp3 via lameenc
```

One voice reads the whole page, code-switching inside the sentence: English prose
in English, the inline Arabic-script terms and the vocalized block quotes recited
natively. A/B the read across models with `ELEVENLABS_MODEL` (multilingual v2 vs
eleven_v3). See [`docs/tts_narration_best_practices.md`](docs/tts_narration_best_practices.md).

## Repo layout

| Path | What |
|---|---|
| `arabize.py` | **Pre-narration step** — deterministic map + Claude Opus for the long tail: Romanized terms → Arabic script (`baseerOutput/` → `arabizedOutput/`) |
| `translit_map.py` | Shared Romanized→Arabic dictionary + honorific-glyph expansion (used by `arabize.py` and `convert_translit.py`) |
| `narrate.py` | **Main pipeline** — text → `audio/full_book_<model>.wav` (ElevenLabs) |
| `baseerOutput/` | Baseer OCR text (arabize.py input) |
| `arabizedOutput/` | Arabic-script script (narrate.py input; generated) |
| `bookImages/` | Source page scans |
| `convert_translit.py` | Romanized → Arabic-script conversion (dictionary; feeds `opusAfterBaseer/`) |
| `opusAfterBaseer/` | Arabic-inline variant of the text (alternative voice path) |
| `transcribe.py` | GPT-4o vision OCR + canonical hadith/Qurʾān resolution (alt/experimental path) |
| `hadith_lookup.py`, `quran_lookup.py` | Canonical text lookup (fawazahmed0 datasets) |
| `baseer_ocr.py`, `opus_ocr.py` | Specialist / Opus OCR fallbacks |
| `benchmark_ocr.py` | GPT-4o vs Baseer OCR benchmark |
| `docs/` | Notes & best-practices |

`audio/` (generated) and `.env` (secrets) are gitignored.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, OPENAI_API_KEY
```

## Usage

```bash
# 1. Convert Romanized Arabic terms to Arabic script (baseerOutput/ → arabizedOutput/)
python arabize.py

# 2. Narrate all pages into one continuous file
python narrate.py                    # → audio/full_book_eleven_multilingual_v2.wav
ELEVENLABS_MODEL=eleven_v3 python narrate.py   # A/B the same text on v3
python narrate.py path/to/textdir    # narrate a different folder

# Convert a rendered WAV to MP3 (128 kbps mono, no ffmpeg)
python - <<'PY'
import wave, lameenc
w = wave.open("audio/full_book_eleven_multilingual_v2.wav","rb"); pcm = w.readframes(w.getnframes())
e = lameenc.Encoder(); e.set_bit_rate(128); e.set_in_sample_rate(w.getframerate())
e.set_channels(1); e.set_quality(2)
open("audio/full_book.mp3","wb").write(e.encode(pcm)+e.flush())
PY
```

Tuning lives at the top of `narrate.py`: `VOICE_SETTINGS`, `FLOW_HONORIFICS_INLINE`,
`CORRECTIONS` (fix Baseer transliteration typos), `MAX_CHARS` (chunk size).

## Notes

- **Numbers** are spelled out in preprocessing so ElevenLabs' own number
  normalizer never runs (it misreads bare section markers and reference numbers).
- **Cost:** ElevenLabs charges ~1 credit/character; the sample 6-page book is
  ~11k credits per full render.
- **Scaling to a full book:** switch the merge unit from *page* to *chapter/section*
  (pages split mid-sentence; chapters don't). The narration logic carries over
  unchanged — only the chunking unit changes.
