#!/usr/bin/env python3
"""convert_translit.py — post-process Baseer OCR output.

Reads each file in ./baseerOutput and writes to ./opusAfterBaseer, converting
ONLY Romanized Arabic terms that carry transliteration accents (macrons ā ī ū,
dotted ḥ ṣ ḍ ṭ ẓ, ʿayn/hamza marks) into bare Arabic script with NO harakat.
Everything else — English prose, the already-vocalized Arabic block quotes,
markdown, honorific glyphs — is left byte-for-byte untouched.

Plain transliterations with no accent (e.g. "Jannah", "Jahannam", "muslim") are
intentionally left as-is, per the "only words with accents" rule.

Patterns are ordered longest-first so multi-word phrases match before their
sub-terms. Diacritic variants Baseer emits (Ṣ vs Ş, ā vs a) are folded with
character classes; apostrophe/ayn/hamza variants match a shared class.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_DIR = Path("baseerOutput")
OUT_DIR = Path("opusAfterBaseer")

# Any apostrophe / ʿayn / hamza transliteration mark.
AP = r"[‘’ʿʾ'`]"
# Sahih skeleton, tolerating Ṣ/Ş and ā/a variants.
SAHIH = r"[ŞṢ][aā][ḥh][īi][ḥh]"

# (regex, arabic) — ORDERED. Longest / most-specific phrases first.
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
    (AP + r"Umar", "عمر"),  # standalone, after the "ibn al-Khattāb" phrase above
    # ---- standalone terms ----
    (r"al-Alb[āa]n[īi]", "الألباني"),
    (r"al-Arna" + AP + r"?[ūu][ṭt]", "الأرناؤوط"),
    (r"id[ṭt][īi]r[āa]b", "اضطراب"),
    (r"Ḥadith", "حديث"),
    (r"wud[ūu]" + AP + r"?", "وضوء"),
    (r"raka" + AP + r"?[āa]t", "ركعات"),
    (r"Rama[ḍd][āa]n", "رمضان"),
    (r"Isl[āa]m", "إسلام"),
    (r"[ŞṢ]uhr", "ظهر"),  # Baseer's mis-transliteration of Ẓuhr
    (SAHIH, "صحيح"),  # standalone Sahih, last
]

def check_article_rule(rules: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Enforce: an Arabic term starting with the definite article ال must come
    from a source pattern that contains 'al-'. Catches ال added to a bare word
    (e.g. Ṣuhr → الظهر), which does not match the English."""
    return [
        (pat, rep) for pat, rep in rules
        if rep.startswith("ال") and not re.search(r"al", pat, re.IGNORECASE)
    ]


# Precompile.
_RULES = [(re.compile(pat), rep) for pat, rep in REPLACEMENTS]


def convert(text: str) -> tuple[str, dict[str, int]]:
    """Apply all replacements; return converted text and per-rule hit counts."""
    hits: dict[str, int] = {}
    for rx, rep in _RULES:
        text, n = rx.subn(rep, text)
        if n:
            hits[rep] = hits.get(rep, 0) + n
    return text, hits


def arabic_lines(text: str) -> list[str]:
    """Lines that are majority Arabic (the block quotes) — for verification."""
    out = []
    for line in text.splitlines():
        letters = [c for c in line if c.strip()]
        arab = [c for c in line if "؀" <= c <= "ۿ"]
        if letters and len(arab) / len(letters) > 0.6:
            out.append(line)
    return out


def main() -> int:
    violations = check_article_rule(REPLACEMENTS)
    if violations:
        print("⚠ article-rule violations (ال added without 'al-' in the English):")
        for pat, rep in violations:
            print(f"    {rep}  ⟵  /{pat}/")
        print()

    OUT_DIR.mkdir(exist_ok=True)
    files = sorted(SRC_DIR.glob("*.txt"))
    print(f"Converting {len(files)} file(s): {SRC_DIR}/ → {OUT_DIR}/\n")

    for src in files:
        text = src.read_text(encoding="utf-8")
        if not text.strip():
            (OUT_DIR / src.name).write_text(text, encoding="utf-8")
            print(f"{src.name}: (empty)")
            continue

        converted, hits = convert(text)
        (OUT_DIR / src.name).write_text(converted, encoding="utf-8")

        # Verify: every pre-existing Arabic block line survives unchanged.
        before, after = arabic_lines(text), converted
        preserved = all(line in after for line in before)
        flag = "" if preserved else "  ⚠ ARABIC BLOCK CHANGED"
        total = sum(hits.values())
        print(f"{src.name}: {total} term(s) converted{flag}")
        for rep, n in hits.items():
            print(f"    {n}× → {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
