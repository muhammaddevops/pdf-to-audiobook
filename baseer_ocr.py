#!/usr/bin/env python3
"""baseer_ocr.py — specialist Arabic OCR fallback for unresolved blocks.

Used only when a hadith block cannot be matched to canonical text (e.g. Musnad
Aḥmad, which the open dataset lacks). Sends the whole page image to the Baseer
(Qwen2.5-VL) endpoint and asks for every fully-vocalized Arabic quotation block,
in reading order, as a JSON array. The caller aligns these to the specific
unresolved placeholders via text containment.

Gated on credentials: if BASEER_API_KEY / BASEER_BASE_URL are unset, this
returns None so the pipeline degrades to "omit + flag" instead of failing.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

BASEER_PROMPT = (
    "You are a faithful Arabic OCR engine. Look at this page image and transcribe "
    "EVERY standalone Arabic quotation block (hadith text, Qurʾānic passages) "
    "EXACTLY as printed, preserving all tashkeel/harakat vowel marks character for "
    "character. Do NOT translate, do NOT correct, do NOT add or drop words or "
    "diacritics. Ignore the English prose and citations. Return ONLY a JSON object "
    'of the form {"blocks": ["<arabic block 1>", "<arabic block 2>", ...]} in '
    "reading order. If there are no Arabic quotation blocks, return "
    '{"blocks": []}.'
)


def _data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None or not mime.startswith("image/"):
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
            image_path.suffix.lower().lstrip("."), "image/png"
        )
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def extract_arabic_blocks(image_path: Path) -> Optional[list[str]]:
    """Return faithful Arabic quote blocks from the page, or None if unavailable.

    None means the fallback could not run (missing credentials or an API error),
    signalling the caller to omit + flag. An empty list means it ran but found no
    Arabic blocks.
    """
    api_key = os.environ.get("BASEER_API_KEY")
    base_url = os.environ.get("BASEER_BASE_URL")
    model = os.environ.get("BASEER_MODEL", "qwen2.5-vl")
    if not api_key or not base_url:
        print(
            "  [baseer] credentials not set (BASEER_API_KEY / BASEER_BASE_URL) — "
            "skipping OCR fallback."
        )
        return None

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": BASEER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(image_path)},
                        },
                    ],
                }
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        # Tolerate a stray code fence around the JSON.
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        blocks = json.loads(raw).get("blocks", [])
        cleaned = [b.strip() for b in blocks if isinstance(b, str) and b.strip()]
        print(f"  [baseer] extracted {len(cleaned)} Arabic block(s) from page.")
        return cleaned
    except Exception as exc:  # missing model, network, bad JSON, etc.
        print(f"  [baseer] fallback failed: {exc}")
        return None
