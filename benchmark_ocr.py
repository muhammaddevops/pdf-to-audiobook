#!/usr/bin/env python3
"""benchmark_ocr.py

Side-by-side OCR benchmark for complex bilingual (Arabic/English) document pages.

The script sends a single local image to two independent vision stacks and writes
their raw transcriptions to separate UTF-8 text files so they can be diffed and
graded manually. It is tuned for three failure modes that break naive OCR:

  1. Romanized Arabic with diacritic macrons  (e.g. Abu Sa'id al-Khudri -> Abū Saʿīd al-Khudrī)
  2. Inline calligraphic honorific glyphs       (e.g. ﷺ  ؓ  ؑ  ﷻ)
  3. Fully vocalized Arabic script (tashkeel)   (harakat sitting above/below letters)

Endpoints:
  * OpenAI stack  -> gpt-4o via the official `openai` SDK
  * Baseer stack  -> baseerocr.com (Qwen2.5-VL). Callable either through the
                     OpenAI-compatible client or a generic HTTP POST, selectable
                     by config so you can swap base URL / token / transport freely.

Environment variables:
  OPENAI_API_KEY        (required for the OpenAI stack)
  BASEER_API_KEY        (required for the Baseer stack)
  BASEER_BASE_URL       (optional, default https://api.baseerocr.com/v1)
  BASEER_MODEL          (optional, default qwen2.5-vl)
  BASEER_TRANSPORT      (optional, "openai" | "http", default "openai")

Usage:
  python benchmark_ocr.py                 # defaults to ./image.png
  python benchmark_ocr.py path/to/page.jpg
"""

from __future__ import annotations

import base64
import mimetypes
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

try:
    from openai import OpenAI
    from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError
except ImportError:  # pragma: no cover - dependency guard
    print(
        "[fatal] The `openai` package is required. Install with: pip install openai requests",
        file=sys.stderr,
    )
    raise


# --------------------------------------------------------------------------- #
# Shared prompt: identical instruction to both stacks for a fair comparison.
# --------------------------------------------------------------------------- #

TRANSCRIPTION_PROMPT: str = (
    "You are a meticulous OCR transcription engine for scholarly bilingual "
    "Arabic/English documents. Transcribe the page EXACTLY as printed. Follow "
    "these rules with zero deviation:\n"
    "1. HONORIFIC GLYPHS: Preserve every Islamic calligraphic honorific as its "
    "exact Unicode character, e.g. ﷺ (U+FDFA), ﷻ (U+FDFB), ؓ (U+0613), "
    "ؑ (U+0611), ؒ (U+0612), ؐ (U+0610). Never romanize, drop, or convert them "
    "into text noise or approximations like '(pbuh)'.\n"
    "2. ROMANIZED ARABIC: Preserve all diacritic macrons and transliteration "
    "marks exactly (ā ī ū ḥ ṣ ḍ ṭ ẓ ʿ ʾ), e.g. 'Abū Saʿīd al-Khudrī'. Do not "
    "strip or normalize them to plain ASCII.\n"
    "3. VOCALIZED ARABIC: Reproduce the full Arabic script INCLUDING every "
    "tashkeel/harakat vowel mark (fatḥa, kasra, ḍamma, sukūn, shadda, tanwīn) "
    "positioned above and below the letters.\n"
    "Maintain the original reading order and line breaks. Output ONLY the raw "
    "transcribed text with no commentary, headers, or code fences."
)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class OcrResult:
    """Outcome of a single OCR endpoint call."""

    stack: str
    text: str = ""
    elapsed_s: float = 0.0
    status: str = "ok"
    status_code: Optional[int] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.text.strip())


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #


def encode_image(image_path: Path) -> tuple[str, str]:
    """Read a local image and return (base64_string, mime_type).

    Raises FileNotFoundError if the path does not exist and ValueError if the
    file type cannot be resolved to a supported image mime type.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None or not mime_type.startswith("image/"):
        # Fall back to PNG for unknown but plausible image extensions.
        suffix = image_path.suffix.lower()
        fallback = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = fallback.get(suffix)
        if mime_type is None:
            raise ValueError(
                f"Unsupported or undetectable image type for: {image_path.name}"
            )

    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return b64, mime_type


def data_url(b64: str, mime_type: str) -> str:
    """Build an RFC 2397 data URL for vision chat endpoints."""
    return f"data:{mime_type};base64,{b64}"


# --------------------------------------------------------------------------- #
# OpenAI GPT-4o stack
# --------------------------------------------------------------------------- #


def transcribe_openai(
    b64: str,
    mime_type: str,
    *,
    model: str = "gpt-4o",
    max_tokens: int = 4096,
    timeout_s: float = 120.0,
) -> OcrResult:
    """Transcribe the image with OpenAI's gpt-4o vision model."""
    stack = "gpt4o"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return OcrResult(
            stack=stack,
            status="error",
            error="Missing OPENAI_API_KEY environment variable.",
        )

    client = OpenAI(api_key=api_key, timeout=timeout_s)
    start = time.perf_counter()
    try:
        print(f"[{stack}] → POST chat/completions  model={model}")
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url(b64, mime_type),
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        elapsed = time.perf_counter() - start
        text = (response.choices[0].message.content or "").strip()
        print(f"[{stack}] ✓ 200 OK in {elapsed:.2f}s  ({len(text)} chars)")
        return OcrResult(
            stack=stack, text=text, elapsed_s=elapsed, status="ok", status_code=200
        )
    except (APITimeoutError, APIConnectionError) as exc:
        elapsed = time.perf_counter() - start
        print(f"[{stack}] ✗ network/timeout after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack, elapsed_s=elapsed, status="error", error=f"network: {exc}"
        )
    except RateLimitError as exc:
        elapsed = time.perf_counter() - start
        print(f"[{stack}] ✗ rate limited after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack,
            elapsed_s=elapsed,
            status="error",
            status_code=429,
            error=f"rate_limit: {exc}",
        )
    except APIError as exc:
        elapsed = time.perf_counter() - start
        code = getattr(exc, "status_code", None)
        print(f"[{stack}] ✗ API error ({code}) after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack,
            elapsed_s=elapsed,
            status="error",
            status_code=code,
            error=f"api_error: {exc}",
        )
    except Exception as exc:  # pragma: no cover - final safety net
        elapsed = time.perf_counter() - start
        print(f"[{stack}] ✗ unexpected error after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack, elapsed_s=elapsed, status="error", error=f"unexpected: {exc}"
        )


# --------------------------------------------------------------------------- #
# Baseer OCR stack (Qwen2.5-VL) — configurable transport
# --------------------------------------------------------------------------- #


@dataclass
class BaseerConfig:
    """Runtime configuration for the Baseer endpoint.

    Everything is env-overridable so the same script targets staging, prod, or a
    self-hosted Qwen2.5-VL gateway without code edits.
    """

    api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("BASEER_API_KEY")
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "BASEER_BASE_URL", "https://api.baseerocr.com/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get("BASEER_MODEL", "qwen2.5-vl")
    )
    transport: str = field(
        default_factory=lambda: os.environ.get("BASEER_TRANSPORT", "openai").lower()
    )
    max_tokens: int = 4096
    timeout_s: float = 120.0


def _baseer_via_openai_client(
    cfg: BaseerConfig, b64: str, mime_type: str
) -> OcrResult:
    """Call Baseer through the OpenAI-compatible chat/completions surface."""
    stack = "baseer"
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_s)
    start = time.perf_counter()
    print(f"[{stack}] → OpenAI-compatible POST {cfg.base_url}  model={cfg.model}")
    response = client.chat.completions.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url(b64, mime_type)},
                    },
                ],
            }
        ],
    )
    elapsed = time.perf_counter() - start
    text = (response.choices[0].message.content or "").strip()
    print(f"[{stack}] ✓ 200 OK in {elapsed:.2f}s  ({len(text)} chars)")
    return OcrResult(
        stack=stack, text=text, elapsed_s=elapsed, status="ok", status_code=200
    )


def _baseer_via_http(cfg: BaseerConfig, b64: str, mime_type: str) -> OcrResult:
    """Call Baseer with a generic HTTP POST payload carrying the base64 image."""
    stack = "baseer"
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url(b64, mime_type)},
                    },
                ],
            }
        ],
    }
    start = time.perf_counter()
    print(f"[{stack}] → raw HTTP POST {url}  model={cfg.model}")
    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)
    elapsed = time.perf_counter() - start
    print(f"[{stack}] ← HTTP {resp.status_code} in {elapsed:.2f}s")
    resp.raise_for_status()
    data = resp.json()
    text = (data["choices"][0]["message"]["content"] or "").strip()
    print(f"[{stack}] ✓ {resp.status_code} OK  ({len(text)} chars)")
    return OcrResult(
        stack=stack,
        text=text,
        elapsed_s=elapsed,
        status="ok",
        status_code=resp.status_code,
    )


def transcribe_baseer(
    b64: str, mime_type: str, cfg: Optional[BaseerConfig] = None
) -> OcrResult:
    """Transcribe the image with the Baseer (Qwen2.5-VL) endpoint.

    Dispatches to the OpenAI-compatible client or the raw HTTP transport based on
    cfg.transport, and normalizes every failure mode into an OcrResult.
    """
    stack = "baseer"
    cfg = cfg or BaseerConfig()

    if not cfg.api_key:
        return OcrResult(
            stack=stack,
            status="error",
            error="Missing BASEER_API_KEY environment variable.",
        )

    dispatch: dict[str, Callable[[BaseerConfig, str, str], OcrResult]] = {
        "openai": _baseer_via_openai_client,
        "http": _baseer_via_http,
    }
    handler = dispatch.get(cfg.transport)
    if handler is None:
        return OcrResult(
            stack=stack,
            status="error",
            error=f"Unknown BASEER_TRANSPORT '{cfg.transport}' (expected 'openai' or 'http').",
        )

    start = time.perf_counter()
    try:
        return handler(cfg, b64, mime_type)
    except requests.exceptions.Timeout as exc:
        elapsed = time.perf_counter() - start
        print(f"[{stack}] ✗ timeout after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack, elapsed_s=elapsed, status="error", error=f"timeout: {exc}"
        )
    except requests.exceptions.HTTPError as exc:
        elapsed = time.perf_counter() - start
        code = exc.response.status_code if exc.response is not None else None
        body = exc.response.text[:500] if exc.response is not None else ""
        print(f"[{stack}] ✗ HTTP {code} after {elapsed:.2f}s: {body}")
        return OcrResult(
            stack=stack,
            elapsed_s=elapsed,
            status="error",
            status_code=code,
            error=f"http_error: {exc}",
        )
    except requests.exceptions.RequestException as exc:
        elapsed = time.perf_counter() - start
        print(f"[{stack}] ✗ network error after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack, elapsed_s=elapsed, status="error", error=f"network: {exc}"
        )
    except (APITimeoutError, APIConnectionError) as exc:
        elapsed = time.perf_counter() - start
        print(f"[{stack}] ✗ client network/timeout after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack, elapsed_s=elapsed, status="error", error=f"network: {exc}"
        )
    except APIError as exc:
        elapsed = time.perf_counter() - start
        code = getattr(exc, "status_code", None)
        print(f"[{stack}] ✗ API error ({code}) after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack,
            elapsed_s=elapsed,
            status="error",
            status_code=code,
            error=f"api_error: {exc}",
        )
    except (KeyError, ValueError) as exc:
        elapsed = time.perf_counter() - start
        print(f"[{stack}] ✗ malformed response after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack,
            elapsed_s=elapsed,
            status="error",
            error=f"bad_response: {exc}",
        )
    except Exception as exc:  # pragma: no cover - final safety net
        elapsed = time.perf_counter() - start
        print(f"[{stack}] ✗ unexpected error after {elapsed:.2f}s: {exc}")
        return OcrResult(
            stack=stack, elapsed_s=elapsed, status="error", error=f"unexpected: {exc}"
        )


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def save_transcript(result: OcrResult, out_path: Path) -> None:
    """Write a transcription to disk with explicit UTF-8 (no lossy fallbacks)."""
    if result.ok:
        out_path.write_text(result.text, encoding="utf-8")
        print(f"[{result.stack}] saved → {out_path}  ({len(result.text)} chars)")
    else:
        note = (
            f"# NO TRANSCRIPT\n"
            f"# stack:  {result.stack}\n"
            f"# status: {result.status}\n"
            f"# code:   {result.status_code}\n"
            f"# error:  {result.error}\n"
        )
        out_path.write_text(note, encoding="utf-8")
        print(f"[{result.stack}] ✗ no transcript — wrote error stub → {out_path}")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main(argv: list[str]) -> int:
    image_path = Path(argv[1]) if len(argv) > 1 else Path("image.png")
    out_dir = image_path.parent if image_path.parent.as_posix() != "" else Path(".")

    print("=" * 70)
    print(f"OCR benchmark  |  image: {image_path}")
    print("=" * 70)

    try:
        b64, mime_type = encode_image(image_path)
        print(f"[encode] {image_path.name} → {mime_type}, {len(b64)} base64 chars")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2

    # Synchronous, sequential runs so the console log reads cleanly per stack.
    print("-" * 70)
    gpt_result = transcribe_openai(b64, mime_type)
    print("-" * 70)
    baseer_result = transcribe_baseer(b64, mime_type)
    print("-" * 70)

    save_transcript(gpt_result, out_dir / "transcript_gpt4o.txt")
    save_transcript(baseer_result, out_dir / "transcript_baseer.txt")

    print("=" * 70)
    print("Summary")
    for r in (gpt_result, baseer_result):
        state = "OK " if r.ok else "ERR"
        print(
            f"  [{state}] {r.stack:<8} {r.elapsed_s:6.2f}s "
            f"code={r.status_code}  {r.error or ''}"
        )
    print("=" * 70)

    # Non-zero exit if either stack failed, useful for CI gating.
    return 0 if (gpt_result.ok and baseer_result.ok) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
