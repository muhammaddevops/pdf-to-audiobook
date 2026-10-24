#!/usr/bin/env python3
"""Render a PDF's pages to bookImages/*.png, ready for the OCR step.

Usage:
    python pdf_to_images.py mybook.pdf                 # all pages at 300 dpi
    python pdf_to_images.py mybook.pdf --first 12 --last 17
    python pdf_to_images.py mybook.pdf --dpi 400 --out bookImages

Pure pip (pypdfium2 bundles its own PDF renderer) — no poppler / system deps.
"""

from __future__ import annotations

import argparse
import pathlib

import pypdfium2 as pdfium


def render(pdf_path: str, out_dir: str, dpi: int, first: int, last: int | None) -> int:
    doc = pdfium.PdfDocument(pdf_path)
    n_pages = len(doc)
    last = min(last or n_pages, n_pages)
    if first < 1 or first > last:
        raise SystemExit(f"page range {first}-{last} is empty (document has {n_pages} pages)")

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72  # PDF user space is 72 dpi

    width = len(str(last))
    for i in range(first - 1, last):
        page = doc[i]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        target = out / f"page{i + 1:0{max(width, 2)}d}.png"
        image.save(target)
        print(f"  {target}  ({image.width}x{image.height})")
    return last - first + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", help="path to the book PDF")
    ap.add_argument("--out", default="bookImages", help="output folder (default: bookImages)")
    ap.add_argument("--dpi", type=int, default=300, help="render resolution (default: 300)")
    ap.add_argument("--first", type=int, default=1, help="first page, 1-based (default: 1)")
    ap.add_argument("--last", type=int, default=None, help="last page, inclusive (default: end)")
    args = ap.parse_args()

    count = render(args.pdf, args.out, args.dpi, args.first, args.last)
    print(f"Wrote {count} page(s) to {args.out}/")


if __name__ == "__main__":
    main()
