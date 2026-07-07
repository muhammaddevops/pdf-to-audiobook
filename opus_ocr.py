#!/usr/bin/env python3
"""opus_ocr.py — Claude Opus 4.8 vision fallback for unresolved Arabic blocks.

Last-resort reader for Arabic quotation blocks that have no canonical source
(e.g. Musnad Aḥmad, or non-canonical Arabic prose) AND that the Baseer specialist
OCR couldn't recover. Runs Opus 4.8 at maximum reasoning effort to transcribe the
page's Arabic quote blocks verbatim, in reading order.

HONEST LIMITATION: Opus is a general vision-language model, like GPT-4o — it is
strong at Arabic but NOT a specialist Arabic OCR, so its reading of dense
vocalized script is best-effort, not definitive. Blocks recovered this way are
marked 'opus' (unverified) in the report so you know their provenance. Use it
only where no canonical text exists to verify against.

Gated on the SDK + credentials: if `anthropic` isn't installed or
ANTHROPIC_API_KEY is unset, this returns None so the pipeline degrades to
"omit + flag" rather than failing.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Optional

try:
    import anthropic
    from anthropic import Anthropic
    _SDK_AVAILABLE = True
except ImportError:  # SDK not installed — fallback stays inert.
    _SDK_AVAILABLE = False

MODEL = "claude-opus-4-8"

OPUS_PROMPT = (
    "You are a meticulous Arabic OCR engine. Look at this page image and "
    "transcribe EVERY standalone Arabic quotation block (hadith text, Qurʾānic "
    "passages, vocalized block quotes) EXACTLY as printed — preserve every "
    "tashkeel/harakat vowel mark character for character. Do NOT translate, "
    "correct, add, or drop words or diacritics; reproduce precisely what the "
    "pixels show. Ignore the English prose and the citations. Return a JSON "
    'object {"blocks": ["<arabic block 1>", ...]} listing the blocks in reading '
    "order; if there are none, return an empty array."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


def _image_block(image_path: Path) -> dict:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None or not mime.startswith("image/"):
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
            image_path.suffix.lower().lstrip("."), "image/png"
        )
    data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": data},
    }


def extract_arabic_blocks(image_path: Path) -> Optional[list[str]]:
    """Return faithful Arabic quote blocks via Opus 4.8, or None if unavailable.

    None means the fallback couldn't run (missing SDK/credentials or an API
    error); the caller then omits + flags. An empty list means it ran but found
    no Arabic blocks.
    """
    if not _SDK_AVAILABLE:
        print("  [opus] anthropic SDK not installed — skipping Opus fallback.")
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [opus] ANTHROPIC_API_KEY not set — skipping Opus fallback.")
        return None

    try:
        client = Anthropic(timeout=600.0)
        # Stream because max-effort adaptive thinking can run long; get the final
        # message once complete. output_config carries both the effort ceiling
        # and the JSON-schema format constraint.
        with client.messages.stream(
            model=MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={"effort": "max", "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OPUS_PROMPT},
                        _image_block(image_path),
                    ],
                }
            ],
        ) as stream:
            message = stream.get_final_message()

        if message.stop_reason == "refusal":
            print("  [opus] request refused by safety classifier — skipping.")
            return None
        raw = next((b.text for b in message.content if b.type == "text"), "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        blocks = json.loads(raw).get("blocks", [])
        cleaned = [b.strip() for b in blocks if isinstance(b, str) and b.strip()]
        print(f"  [opus] extracted {len(cleaned)} Arabic block(s) from page.")
        return cleaned
    except anthropic.AuthenticationError:
        print("  [opus] authentication failed — check ANTHROPIC_API_KEY.")
    except anthropic.APIConnectionError as exc:
        print(f"  [opus] network error — skipping: {exc}")
    except anthropic.APIStatusError as exc:
        print(f"  [opus] API error ({exc.status_code}) — skipping: {exc}")
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"  [opus] malformed response — skipping: {exc}")
    except Exception as exc:  # final safety net — never break the pipeline
        print(f"  [opus] fallback failed — skipping: {exc}")
    return None
