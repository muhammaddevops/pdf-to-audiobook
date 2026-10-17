# Plan: Read transliterated words natively in Arabic

**Goal:** the prospect wants every English *transliterated* Arabic term (e.g. `Ṣaḥīḥ`,
`Sunan al-Tirmidhī`, `Ẓuhr`, `Abū Ayyūb`) **pronounced natively in Arabic** in the
narration, instead of the current anglicized reading.

**Status:** planning / not started. Demo of 5 pages got good prospect feedback;
this is the feature ask to move forward.
**Owner:** —  ·  **Last updated:** 2026-07-05

---

## Where things stand

- Pipeline: Baseer OCR → [`narrate.py`](../narrate.py) `preprocess()` → ElevenLabs
  (one call/page) → `audio/`.
- Current model **`eleven_multilingual_v2`**, single voice **Daniel**
  (`onwK4e9ZLuTAKqWW03F9`).
- **Today `narrate.py` does the *opposite* of the ask:** `fold_latin()` strips the
  diacritics so Daniel reads terms anglicized (`al-Tirmidhī` → "al-Tirmidhi").
  Going native means *replacing* those terms with Arabic script, not folding them.
- Pacing work already done (keep): honorifics read inline, citation comma lightened
  — see [`docs/tts_narration_best_practices.md`](./tts_narration_best_practices.md).
- A print-only Arabic variant exists as a reference for conversion output:
  [`output/opus_transcript_ar.txt`](../output/opus_transcript_ar.txt).
- A term→Arabic map is being started in [`convert_translit.py`](../convert_translit.py).

## The core constraint (read this first)

**No ElevenLabs model does reliable mid-sentence single-word code-switching** — an
isolated Arabic term inside an English sentence is the hardest case.
- `multilingual_v2`: language detected once per call, no mid-call switch; English
  voices carry English phonetics (accent bleed). Phoneme/IPA control is English-only
  and unsupported here.
- `eleven_v3` (newest, GA): **does** code-switch in one generation *and keeps one
  voice profile* — big improvement — but clean switches are between sentences;
  mid-sentence is "hit or miss." Loses SSML `<break>` tags; 5k char limit; less stable.

So "native Arabic + smooth single narrator" is achievable **only if v3's in-voice
Arabic holds up on isolated terms** — which must be tested, not assumed.

## Two independent decisions

### Decision 1 — Rendering strategy

| # | Approach | Native-ness | One narrator | Seams | Effort |
|---|---|---|---|---|---|
| 1 | Inline Arabic, **v3**, single voice | Med–High | ✅ | switch blips | small |
| 2 | Inline Arabic, Daniel + multilingual v2 | Low (bleed) | ✅ | blips | tiny |
| 3 | Split by language, **native Arabic voice**, stitch | **Highest** | ❌ 2 people | seam/term | high |
| 4 | Split by language, same voice both segments | Med | ✅ | seam/term | high |
| 5 | Phonetic Latin respelling (no Arabic script) | Low–Med (approx) | ✅ | none | small |
| 6 | Whole book by a bilingual Arabic-native voice | High Arabic / accented English | ✅ | none | med |

Note: `narrate.py` already concatenates PCM per chunk, so splice plumbing for 3/4 is
a small lift — the hard part is seam prosody, not code.

### Decision 2 — Transliteration → Arabic conversion (needed for 1–4, 6)

- **Hybrid recommended:** curated map ([`convert_translit.py`](../convert_translit.py))
  → LLM fallback for unknown terms → verification pass.
- **Harakat:** feed **fully voweled** Arabic to TTS (unvoweled = ambiguous =
  mispronounced), even if the print copy uses minimal harakat. Audio and print may
  need different harakat levels.
- **Unit-level trigger:** convert whole name/title units, not just the accented word
  (`Sunan al-Tirmidhī` → all of `سنن الترمذي`, not `Sunan` + `الترمذي`).

## Recommendation

1. **Probe before building.** One page, ~5 terms, voweled Arabic, generated three ways:
   (1) **v3 single voice inline** ← most likely clean win, (2) multilingual v2 inline
   (baseline), (6) a bilingual Arabic-native voice. Judge by ear. Cheap.
2. If v3 in-voice Arabic holds up on isolated terms → ship that (one voice, native,
   no seams). If not → fall back to **stitching a native Arabic voice** (3/4).
3. Consider a **per-page hybrid**: v3 for Arabic-dense pages, multilingual v2 (with
   our break-tag pacing) for English-only pages.

## Next steps (pick up here)

- [ ] Confirm the 3 open questions with the prospect (below).
- [ ] Pick 1 test page + list its ~5 transliterated terms with correct **voweled**
      Arabic (hand-check hamza seats: `السائب`, `الأرناؤوط`).
- [ ] Choose a candidate **native/bilingual Arabic voice** in ElevenLabs for options 1/6.
- [ ] Build a throwaway probe script (don't touch `narrate.py` yet) that renders the
      page via v3 / multilingual v2 / bilingual voice → 3 wav files.
- [ ] Listen; decide rendering strategy from Decision 1.
- [ ] Only then: wire chosen strategy into `narrate.py` (+ conversion in
      `convert_translit.py`), and re-tune pacing (v3 has no `<break>` tags).

## Open questions for the prospect

1. **Narrator identity:** one consistent voice required, or is a second native-Arabic
   voice for the Arabic terms acceptable? (Decides #3 vs the rest.)
2. **"Native" bar:** true Arabic phonemes (ع/ح/ظ/ق — needs native voice) or just
   "clearly Arabic, not anglicized" (a bilingual voice / phonetic respell may pass)?
3. **Scope:** only technical terms/titles, or also common words already in
   `convert_translit.py` like `Islam→إسلام`, `Allah→الله`? Denser Arabic = worse
   seams/bleed; this changes which option wins.

## References

- [Models — ElevenLabs](https://elevenlabs.io/docs/overview/models)
- [Eleven v3](https://elevenlabs.io/v3)
- [Pronunciation dictionaries — ElevenLabs](https://elevenlabs.io/docs/eleven-api/guides/how-to/text-to-speech/pronunciation-dictionaries)
- [Multilingual voice-agent limitations — Deepgram](https://deepgram.com/learn/using-elevenlabs-multilingual-voice-agents-limitations)
- [Languages vs Accents (2026) — Deepgram](https://deepgram.com/learn/elevenlabs-languages-vs-accents-support)
- Internal: [`tts_narration_best_practices.md`](./tts_narration_best_practices.md)
