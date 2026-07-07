# TTS Narration Best Practices (ElevenLabs — Daniel / `eleven_multilingual_v2`)

Reference for keeping the narration flowing and pausing only where it should.
Our pipeline is [`narrate.py`](../narrate.py): Baseer OCR text → `preprocess()` →
one ElevenLabs call per page (multilingual v2, Daniel voice `onwK4e9ZLuTAKqWW03F9`).

## How the model decides where to pause

`eleven_multilingual_v2` has **no prosody model that reads meaning** — it infers
pauses almost entirely from **punctuation and structure**. Rough pause lengths,
shortest → longest:

| Marker | Effect |
|---|---|
| `—` / `--` (dash) | short pause |
| `,` (comma) | short pause |
| `…` / `...` (ellipsis) | weightier, hesitant pause |
| `.` `?` `!` (sentence end) | longer pause |
| line break (`\n`) | pause |
| blank line / paragraph (`\n\n`) | longer pause |
| `( … )` parentheses | **pause on entry and exit** — a parenthetical is set off audibly |

**Key consequence:** written text is punctuated for *reading* (grammar, citations,
appositives). A human narrator pauses far less than the comma count implies. The
job of preprocessing is to drop the *grammatical-but-not-prosodic* breaks and keep
only the ones a storyteller would actually take.

The two worst offenders in *our* text:

1. **Parenthetical honorifics.** Every `(ﷺ)` / `(رضي الله عنه)` becomes
   `(peace be upon him)` / `(may Allah be pleased with him)` — a bracketed
   parenthetical that Daniel pauses around, several times per page, mid-sentence.
   In recitation these set phrases are said *inline*, in one breath with the name.
2. **Citations.** `(Sunan al-Tirmidhi 478, Sahih)` stacks a parenthetical **and**
   an internal comma at the end of every quote → a little stutter.

## Explicit pause control: SSML break tags

Break tags give deterministic control instead of relying on punctuation inference:

```
<break time="1.0s" />
```

- **Supported on v2-family models (incl. `eleven_multilingual_v2` — our model).**
  **NOT supported on Eleven v3** (v3 uses audio tags / ellipses instead).
- Max **3 seconds** per tag.
- ⚠️ **Use sparingly.** "Using too many break tags in a single generation can
  cause instability. The AI might speed up, or introduce additional noises or
  audio artifacts." A handful per page is fine; one after every honorific is not.

Because our honorifics are frequent, we prefer *removing* the paren-pause over
*adding* break tags — fewer tags, less artifact risk.

## Voice settings for long-form narration

Current settings in `narrate.py` (`VOICE_SETTINGS`) and guidance:

| Setting | Ours | Guidance |
|---|---|---|
| `stability` | 0.50 | Balanced. Higher → flatter, more consistent, less expressive; lower → more emotional but variable. 0.45–0.55 is a good narration band. Does **not** change *where* it pauses — that's punctuation. |
| `similarity_boost` | 0.80 | Keep high (0.75–0.85) to stay on-voice. |
| `style` | 0.15 | Keep low for even narration; high style exaggerates and destabilises. |
| `speed` | 0.96 | Range **0.7–1.2** (extremes hurt quality). 0.96 is a measured pace; 1.0 flows a touch more if it feels draggy. |
| `use_speaker_boost` | true | Fine to leave on. |

## Cross-chunk prosody

When a page is split into multiple requests, pass `previous_text` / `next_text`
(or `previous_request_ids`) so prosody carries across the seam. `narrate.py`
already does this — keep it.

## Text normalization

Normalization (numbers, dates) is **on by default** for all TTS models;
multilingual v2 "can better generalize the reading out of numbers in a way that
is more natural." No action needed for hadith reference numbers.

## Audiobook narration principles (general)

- Punctuation in the *script fed to TTS* should map to **breath / sense units**,
  not written-grammar rules. Remove commas that a narrator wouldn't honor.
- Don't blanket-strip commas — that removes *meaningful* pauses and can garble
  clause boundaries. Target known choppy patterns (citations, set-off phrases).
- Set phrases (honorifics, formulae) read inline, not as asides.
- Always A/B a single page by ear before regenerating the whole book — pacing is
  judged by listening, not by reading the script.

## What `narrate.py` now applies (`smooth_pacing()`)

Applied after honorific expansion + Latin folding, before chunking. Each rule is
a toggle at the top of the file so you can A/B:

- `FLOW_HONORIFICS_INLINE` — drop the parentheses around expanded honorifics so
  they blend into the sentence (`the Prophet peace be upon him would…`), removing
  the biggest source of mid-sentence pausing.
- `THIN_CITATION_COMMAS` — inside a citation, replace the comma before the grade
  with an em dash (`(Sunan al-Tirmidhi 478 — Sahih)`), lightening the end-of-quote
  stutter while keeping a clean separation.

Both are conservative and reversible. Tune by regenerating one page and listening.

---

### Sources

- [Best practices — ElevenLabs Docs](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices)
- [How can I add pauses? — ElevenLabs Help](https://help.elevenlabs.io/hc/en-us/articles/13416374683665-How-can-I-add-pauses)
- [Speed control — ElevenLabs Docs](https://elevenlabs.io/docs/agents-platform/customization/voice/speed-control)
- [Audiobooks — ElevenLabs Docs](https://elevenlabs.io/docs/eleven-creative/products/audiobooks)
- [ElevenLabs Cheat Sheet (2026)](https://www.webfuse.com/elevenlabs-cheat-sheet)
