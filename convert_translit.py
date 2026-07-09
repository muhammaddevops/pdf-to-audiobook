#!/usr/bin/env python3
"""convert_translit.py — deterministic Baseer OCR → Arabic-script (dictionary).

Reads each file in ./baseerOutput and writes to ./opusAfterBaseer, converting
Romanized Arabic terms to bare Arabic script using the shared dictionary in
translit_map.py (the same map arabize.py runs first). English prose, the
already-vocalized Arabic block quotes, markdown, and honorific glyphs are left
untouched. This is the pure-dictionary path; arabize.py layers an LLM pass on top
for the long tail.
"""

from __future__ import annotations

from pathlib import Path

from translit_map import apply_map, arabic_lines, check_article_rule

SRC_DIR = Path("baseerOutput")
OUT_DIR = Path("opusAfterBaseer")


def main() -> int:
    violations = check_article_rule()
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

        converted, hits = apply_map(text)
        (OUT_DIR / src.name).write_text(converted, encoding="utf-8")

        # Verify: every pre-existing Arabic block line survives unchanged.
        preserved = all(line in converted for line in arabic_lines(text))
        flag = "" if preserved else "  ⚠ ARABIC BLOCK CHANGED"
        total = sum(hits.values())
        print(f"{src.name}: {total} term(s) converted{flag}")
        for rep, n in hits.items():
            print(f"    {n}× → {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
