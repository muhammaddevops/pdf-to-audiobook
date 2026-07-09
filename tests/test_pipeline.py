#!/usr/bin/env python3
"""Objective tests for the text pipeline (no API keys / network needed).

Covers the deterministic arabization (translit_map), the narrate preprocessing,
and the page-merge logic — the parts we can score without listening. The
ElevenLabs read quality is judged by ear / by ab_score.py, not here.

Run:  pytest -q
"""

from __future__ import annotations

import unicodedata

import translit_map as tm
import narrate

# --------------------------------------------------------------------------- #
# Honorifics — expanded to Arabic words, brackets dropped, no glyphs left.
# --------------------------------------------------------------------------- #

def test_prophetic_glyph_expands_to_arabic():
    out = tm.expand_honorifics("The Messenger of Allah (ﷺ) said")
    assert "صلى الله عليه وسلم" in out
    assert "ﷺ" not in out
    assert "(صلى الله عليه وسلم)" not in out  # brackets dropped for flow


def test_companion_glyphs_expand():
    for src in ["Abu Hurayrah (ؓ) reported", "Abu Hurayrah () reported"]:
        out = tm.expand_honorifics(src)
        assert "رضي الله عنه" in out
        assert "ؓ" not in out and "" not in out


def test_written_companion_honorific_kept_arabic():
    # Already-Arabic honorific must stay Arabic (never flip back to English).
    out = tm.expand_honorifics("(رضي الله عنه)")
    assert out.strip() == "رضي الله عنه"


def test_no_honorific_glyphs_survive_sample():
    raw = open("baseerOutput/page12.txt", encoding="utf-8").read()
    out = tm.expand_honorifics(raw)
    assert not any(g in out for g in ["ﷺ", "ﷻ", "ؓ", ""])


# --------------------------------------------------------------------------- #
# Term map — known citation terms and common vocabulary convert to Arabic.
# --------------------------------------------------------------------------- #

def test_citation_terms_convert():
    out, _ = tm.apply_map("reported in Sunan al-Tirmidhī, graded Ṣaḥīḥ")
    assert "سنن الترمذي" in out
    assert "صحيح" in out


def test_common_vocabulary_converts():
    cases = {
        "prayed Fajr": "فجر", "the Zuhr prayer": "ظهر", "went for Hajj": "حج",
        "gates of Jannah": "جنة", "the Prophet Muhammad": "محمد",
        "recited the Qurʾān": "قرآن", "a Surah": "سورة",
    }
    for src, expected in cases.items():
        out, _ = tm.apply_map(src)
        assert expected in out, f"{src!r} -> {out!r}"


def test_no_midword_collisions():
    # English words that merely CONTAIN an Arabic-looking substring stay intact.
    for word in ["Islamic", "visa", "Nepali", "Somalia", "Alice"]:
        out, _ = tm.apply_map(f"an {word} example")
        assert word in out, f"{word!r} was clobbered -> {out!r}"


def test_standalone_islam_still_converts():
    out, _ = tm.apply_map("the religion of Islam")
    assert "إسلام" in out


# --------------------------------------------------------------------------- #
# Invariants: no harakat added, existing Arabic blocks preserved, map clean.
# --------------------------------------------------------------------------- #

def test_map_output_has_no_harakat():
    # Everything the map GENERATES must be bare (no combining marks).
    for _pat, arabic in tm.REPLACEMENTS:
        assert not any(unicodedata.category(c) == "Mn" for c in arabic), arabic


def test_existing_arabic_block_preserved():
    raw = open("baseerOutput/page12.txt", encoding="utf-8").read()
    out, _ = tm.apply_map(tm.expand_honorifics(raw))
    for line in tm.arabic_lines(raw):
        assert line in out, "a vocalized block line was altered"


def test_article_rule_is_clean():
    assert tm.check_article_rule() == []


def test_all_sample_pages_are_map_only():
    import glob
    for f in sorted(glob.glob("baseerOutput/*.txt")):
        raw = open(f, encoding="utf-8").read()
        if not raw.strip():
            continue
        out, _ = tm.apply_map(tm.expand_honorifics(raw))
        assert not tm.has_translit_residue(out), f"{f} left transliteration residue"


# --------------------------------------------------------------------------- #
# Residue detector — drives the LLM short-circuit.
# --------------------------------------------------------------------------- #

def test_residue_true_on_unknown_accented_term():
    assert tm.has_translit_residue("al-Bukhārī narrated")  # not in the map


def test_residue_false_on_plain_and_arabic():
    assert not tm.has_translit_residue("plain English and محمد")


# --------------------------------------------------------------------------- #
# narrate.py preprocessing.
# --------------------------------------------------------------------------- #

def test_preprocess_spells_numbers_and_strips_markdown():
    out = narrate.preprocess("**Chapter 3** has 12 sections")
    assert "*" not in out
    assert not any(ch.isdigit() for ch in out)
    assert "three" in out and "twelve" in out


def test_preprocess_keeps_arabic_and_expands_honorific():
    out = narrate.preprocess("The Prophet (ﷺ) said الحمد لله")
    assert "صلى الله عليه وسلم" in out
    assert "ﷺ" not in out
    assert "الحمد لله" in out  # existing Arabic untouched


def test_number_to_words():
    assert narrate.number_to_words(0) == "zero"
    assert narrate.number_to_words(478) == "four hundred seventy-eight"
    assert narrate.number_to_words(2023) == "two thousand twenty-three"


def test_build_document_joins_midsentence_page_seam(tmp_path):
    a = tmp_path / "page1.txt"
    b = tmp_path / "page2.txt"
    a.write_text("A sentence that runs", encoding="utf-8")           # no terminal punct
    b.write_text("into the next page.", encoding="utf-8")            # not a new section
    merged = narrate.build_document([a, b])
    assert "runs into the next" in merged  # joined across the seam


def test_build_document_keeps_new_section_break(tmp_path):
    a = tmp_path / "page1.txt"
    b = tmp_path / "page2.txt"
    a.write_text("End of a thought", encoding="utf-8")
    b.write_text("1. A new numbered section", encoding="utf-8")
    merged = narrate.build_document([a, b])
    assert "thought\n\n1. A new" in merged  # break preserved
