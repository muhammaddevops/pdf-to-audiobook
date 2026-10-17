#!/usr/bin/env python3
"""translit_map.py — deterministic Romanized-Arabic → Arabic-script map.

The single source of truth for turning transliterated Arabic terms into bare
(unvocalized) Arabic script before TTS, plus expanding honorific glyphs to their
Arabic words. Runs first in arabize.py; only the terms it can't resolve go to the
LLM. convert_translit.py imports the same map so both paths stay in sync.

Design rules (per project decisions):
  * Output is BARE Arabic — no harakat. The TTS voice supplies the vowels;
    injected tashkeel only adds pronunciation-error risk.
  * Existing vocalized Arabic block quotes are never touched (this only rewrites
    Latin-script terms and honorific glyphs).
  * Honorifics are read natively in Arabic: the ﷺ / companion ligature glyphs are
    unreliable in TTS, so they're expanded to their Arabic words.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Honorifics — expand glyphs to bare Arabic words, read natively by the voice.
# --------------------------------------------------------------------------- #

# The single-codepoint ligatures / signs the source uses, keyed by codepoint so
# the invisible / private-use ones stay unambiguous. TTS engines handle these
# inconsistently (skip or mangle), so we expand them to Arabic words.
_HONORIFIC_GLYPHS = {
    "ﷺ": "صلى الله عليه وسلم",  # U+FDFA ligature: sallallahu alayhi wasallam
    "ﷻ": "جل جلاله",            # U+FDFB ligature: jalla jalaluhu
    "ؓ": "رضي الله عنه",         # U+0613 sign: radi allahu anhu
    "": "رضي الله عنه",         # Baseer private-use glyph that renders empty
}

# Already-Arabic honorific phrases the source prints inside brackets. Bracketing
# makes the voice pause on entry/exit mid-sentence, so we drop the parens and let
# the set phrase blend in — the phrase itself is left untouched.
_HONORIFIC_PHRASES = ("صلى الله عليه وسلم", "جل جلاله", "رضي الله عنه")


def expand_honorifics(text: str) -> str:
    """Expand honorific glyphs to bare Arabic words and de-bracket set phrases.

    Idempotent: on text that's already been expanded (no glyphs left) it only
    strips any remaining brackets around the Arabic phrases.
    """
    for glyph, phrase in _HONORIFIC_GLYPHS.items():
        # "(glyph)" → phrase (drop the pausing parentheses), then the bare glyph.
        text = re.sub(rf"\(\s*{re.escape(glyph)}\s*\)", phrase, text)
        text = text.replace(glyph, phrase)
    # Written companion honorific as Baseer emits it (bare Arabic, gender/plural).
    text = re.sub(r"رض[يى]\s+الله\s+عنه(?:ما|م|ا)?", "رضي الله عنه", text)
    # Drop parentheses that wrap only an honorific phrase so it reads inline.
    for phrase in _HONORIFIC_PHRASES:
        text = re.sub(rf"\(\s*({re.escape(phrase)})\s*\)", r"\1", text)
    return text


# --------------------------------------------------------------------------- #
# Term map — Romanized Arabic-origin terms → bare Arabic script.
# Ordered longest / most-specific first so multi-word phrases match before their
# sub-terms. Diacritic variants Baseer emits (Ṣ/Ş, ā/a) are folded with classes.
# --------------------------------------------------------------------------- #

# Any apostrophe / ʿayn / hamza transliteration mark.
AP = r"[‘’ʿʾ'`]"
# Sahih skeleton, tolerating Ṣ/Ş and ā/a variants.
SAHIH = r"[ŞṢ][aā][ḥh][īi][ḥh]"

REPLACEMENTS: list[tuple[str, str]] = [
    # ---- multi-word citation titles ----
    (r"Tuhfat al-A[ḥh]wadhi Shar[ḥh] Sunan al-Tirmidh[īi]", "تحفة الأحوذي شرح سنن الترمذي"),
    (SAHIH + r" al-J[āa]mi" + AP, "صحيح الجامع"),
    (SAHIH + r" Ibn [ḤH]ibb[āa]n", "صحيح ابن حبان"),
    (SAHIH + r" Muslim", "صحيح مسلم"),
    (SAHIH + r" li-ghayrih[īi]", "صحيح لغيره"),
    (r"Sunan al-Tirmidh[īi]", "سنن الترمذي"),
    (r"Sunan Ab[ūu] D[āa]w[ūu]d", "سنن أبي داود"),
    (r"Musnad A[ḥh]mad", "مسند أحمد"),
    (r"Ab[ūu] " + AP + r"?Is[āa] al-Tirmidh[īi]", "أبو عيسى الترمذي"),
    # ---- narrator / scholar names (before standalone terms) ----
    (AP + r"Umar ibn al-Khatt[āa]b", "عمر بن الخطاب"),
    (AP + r"Abdull[āa]h ibn al-S[āa]" + AP + r"ib", "عبد الله بن السائب"),
    (AP + r"Uqbah ibn " + AP + r"?[ĀA]mir", "عقبة بن عامر"),
    (r"Al-Bar[āa]" + AP + r" ibn " + AP + r"?[ĀA]zib", "البراء بن عازب"),
    (r"J[āa]bir ibn " + AP + r"?Abdill[āa]h", "جابر بن عبد الله"),
    (r"Ab[ūu] Hurayrah", "أبو هريرة"),
    (r"Ab[ūu] Ayy[ūu]b", "أبو أيوب"),
    (AP + r"Umar", "عمر"),  # standalone, after the "ibn al-Khattab" phrase above
    # ---- standalone terms (accented) ----
    (r"al-Alb[āa]n[īi]", "الألباني"),
    (r"al-Arna" + AP + r"?[ūu][ṭt]", "الأرناؤوط"),
    (r"id[ṭt][īi]r[āa]b", "اضطراب"),
    (r"Ḥadith", "حديث"),
    (r"wud[ūu]" + AP + r"?", "وضوء"),
    (r"raka" + AP + r"?[āa]t", "ركعات"),
    (r"Rama[ḍd][āa]n", "رمضان"),
    (r"\bIsl[āa]m\b", "إسلام"),
    (r"[ŞṢ]uhr", "ظهر"),  # Baseer's mis-transliteration of Zuhr
    (SAHIH, "صحيح"),  # standalone Sahih, last
]

# --------------------------------------------------------------------------- #
# Common Arabic vocabulary — the words that recur across most Islamic books.
# This is the general-purpose core (NOT book-specific); grow it over time. Each
# entry is case-insensitive ((?i)) and tolerant of the usual transliteration
# variants (ā/a, ī/i, ū/u, and an optional leading ʿayn/hamza via AP). Bare
# Arabic output, no harakat. Longer/compound terms come before their sub-words.
# --------------------------------------------------------------------------- #
_C = r"(?i)"  # case-insensitive prefix

COMMON: list[tuple[str, str]] = [
    # ---- the five daily prayers ----
    (_C + r"\bFajr\b", "فجر"),
    (_C + r"\bDhuhr\b", "ظهر"),
    (_C + r"\b[DZ]uhr\b", "ظهر"),                       # Zuhr / Duhr
    (_C + r"\b" + AP + r"?[ĀA]sr\b", "عصر"),
    (_C + r"\bMaghrib\b", "مغرب"),
    (_C + r"\b" + AP + r"?Ish[āa]" + AP + r"?\b", "عشاء"),   # Isha / ʿIshāʾ
    # ---- pillars & acts of worship ----
    (_C + r"\b[ŞṢ]al[āa][ht]\b", "صلاة"),               # salah / salat / ṣalāh
    (_C + r"\bZak[āa][ht]\b", "زكاة"),
    (_C + r"\b[ŞṢ]awm\b", "صوم"),
    (_C + r"\b[ḤH]ajj\b", "حج"),
    (_C + r"\b" + AP + r"?Umrah\b", "عمرة"),
    (_C + r"\bAdh[āa]n\b", "أذان"),
    (_C + r"\bIq[āa]mah\b", "إقامة"),
    (_C + r"\bSuj[ūu]d\b", "سجود"),
    (_C + r"\bRuk[ūu]" + AP + r"?\b", "ركوع"),
    (_C + r"\bRak" + AP + r"?ah\b", "ركعة"),
    (_C + r"\bTaraw[īi][ḥh]\b", "تراويح"),
    # ---- core concepts ----
    (_C + r"\b[īi]m[āa]n\b", "إيمان"),
    (_C + r"\bTaqw[āa]\b", "تقوى"),
    (_C + r"\bTaw[ḥh][īi]d\b", "توحيد"),
    (_C + r"\bShirk\b", "شرك"),
    (_C + r"\bKufr\b", "كفر"),
    (_C + r"\bSunnah\b", "سنة"),
    (_C + r"\bBid" + AP + r"?ah\b", "بدعة"),
    (_C + r"\bFar[ḍd]\b", "فرض"),
    (_C + r"\b[ḤH]al[āa]l\b", "حلال"),
    (_C + r"\b[ḤH]ar[āa]m\b", "حرام"),
    (_C + r"\bMakr[ūu]h\b", "مكروه"),
    (_C + r"\bMusta[ḥh]abb\b", "مستحب"),
    (_C + r"\bBarakah\b", "بركة"),
    (_C + r"\bRa[ḥh]mah\b", "رحمة"),
    (_C + r"\bJih[āa]d\b", "جهاد"),
    (_C + r"\bDu" + AP + r"[āa]?\b", "دعاء"),           # dua / duʿāʾ
    (_C + r"\bDhikr\b", "ذكر"),
    (_C + r"\bTawbah\b", "توبة"),
    (_C + r"\b[ŞṢ]abr\b", "صبر"),
    (_C + r"\bShukr\b", "شكر"),
    # ---- Qurʾān & scholarship ----
    (_C + r"\bQur" + AP + r"?[āa]n\b", "قرآن"),
    (_C + r"\bS[ūu]rah\b", "سورة"),
    (_C + r"\b[āA]yah\b", "آية"),
    (_C + r"\bTafs[īi]r\b", "تفسير"),
    (_C + r"\bTajw[īi]d\b", "تجويد"),
    # ---- places ----
    (_C + r"\bMasjid\b", "مسجد"),
    (_C + r"\bQiblah?\b", "قبلة"),
    (_C + r"\bKa" + AP + r"?bah\b", "كعبة"),
    (_C + r"\bMakkah\b", "مكة"),
    (_C + r"\bMecca\b", "مكة"),
    # ---- prophets, angels, well-known figures ----
    (_C + r"\bMu[ḥh]ammad\b", "محمد"),
    (_C + r"\bRas[ūu]l\b", "رسول"),
    (_C + r"\bNab[īi]\b", "نبي"),
    (_C + r"\bIbr[āa]h[īi]m\b", "إبراهيم"),
    (_C + r"\bM[ūu]s[āa]\b", "موسى"),
    (_C + r"\b" + AP + r"?[īI]s[āa]\b", "عيسى"),
    (_C + r"\bY[ūu]suf\b", "يوسف"),
    (_C + r"\bJibr[īi]l\b", "جبريل"),
    (_C + r"\bAb[ūu] Bakr\b", "أبو بكر"),
    (_C + r"\b" + AP + r"?Uthm[āa]n\b", "عثمان"),
    (_C + r"\b" + AP + r"?Al[īi]\b", "علي"),
    (_C + r"\b" + AP + r"?[āA]" + AP + r"?ishah\b", "عائشة"),
    (_C + r"\bF[āa][ṭt]imah\b", "فاطمة"),
    (_C + r"\bBil[āa]l\b", "بلال"),
    # ---- paradise / hellfire (kept from the seed list) ----
    (_C + r"\bJannah\b", "جنة"),
    (_C + r"\bJahannam\b", "جهنم"),
]

REPLACEMENTS += COMMON

# Precompile.
_RULES = [(re.compile(pat), rep) for pat, rep in REPLACEMENTS]


def apply_map(text: str) -> tuple[str, dict[str, int]]:
    """Apply all term replacements; return converted text and per-rule hit counts."""
    hits: dict[str, int] = {}
    for rx, rep in _RULES:
        text, n = rx.subn(rep, text)
        if n:
            hits[rep] = hits.get(rep, 0) + n
    return text, hits


# --------------------------------------------------------------------------- #
# Residue detection — is there still a Latin-transliterated Arabic term left?
# --------------------------------------------------------------------------- #

_AYN_HAMZA = {"ʿ", "ʾ", "ʼ", "ʹ", "ʻ"}


def has_translit_residue(text: str) -> bool:
    """True if any Latin letter still carries a transliteration diacritic.

    A macron/dot/ayn mark on a Latin letter (ā, ī, ḥ, ṣ, ʿ …) means an
    Arabic-origin term the map didn't convert — send the page to the LLM. Plain
    ASCII English is ignored. NOTE: unaccented transliterations the map misses
    (a bare name) won't be flagged — those are handled by adding them to the map.
    """
    for ch in text:
        if ch in _AYN_HAMZA:
            return True
        if unicodedata.name(ch, "").startswith("LATIN"):
            if any(unicodedata.combining(c) for c in unicodedata.normalize("NFKD", ch)):
                return True
    return False


# --------------------------------------------------------------------------- #
# Verification helper (shared with convert_translit.py).
# --------------------------------------------------------------------------- #

def arabic_lines(text: str) -> list[str]:
    """Lines that are majority Arabic (the pre-existing block quotes)."""
    out = []
    for line in text.splitlines():
        letters = [c for c in line if c.strip()]
        arab = [c for c in line if "؀" <= c <= "ۿ"]
        if letters and len(arab) / len(letters) > 0.6:
            out.append(line.strip())
    return out


def check_article_rule(rules: list[tuple[str, str]] = REPLACEMENTS) -> list[tuple[str, str]]:
    """Flag rules that add the definite article ال without an 'al-' in the source
    pattern (e.g. a bare Suhr → الظهر), which wouldn't match the English."""
    return [
        (pat, rep) for pat, rep in rules
        if rep.startswith("ال") and not re.search(r"al", pat, re.IGNORECASE)
    ]
