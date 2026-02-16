"""
Token Packer Utilities
======================

Lightweight helpers to:
- count tokens (approx via tiktoken)
- truncate text by token budget (not by characters)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tiktoken


def _get_encoder(model_hint: Optional[str] = None):
    if model_hint:
        try:
            return tiktoken.encoding_for_model(model_hint)
        except Exception:
            pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Offline/no-cache fallback.
        return None


def count_tokens(text: str, model_hint: Optional[str] = None) -> int:
    enc = _get_encoder(model_hint)
    raw = text or ""
    if enc is not None:
        try:
            return len(enc.encode(raw))
        except Exception:
            pass
    return max(1, int(len(raw) / 4)) if raw else 0


def truncate_text_by_tokens(
    text: str,
    max_tokens: int,
    model_hint: Optional[str] = None,
    suffix: str = "...[truncated]",
) -> str:
    """
    Truncate `text` to <= `max_tokens` tokens, appending `suffix` when truncated.
    """
    if text is None:
        return ""
    if max_tokens <= 0:
        return ""

    enc = _get_encoder(model_hint)
    if enc is not None:
        try:
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text

            # Leave room for suffix tokens.
            suffix_tokens = enc.encode(suffix) if suffix else []
            budget = max(0, max_tokens - len(suffix_tokens))
            truncated = enc.decode(tokens[:budget])
            return truncated + (suffix if suffix else "")
        except Exception:
            pass

    # Heuristic char-based fallback (~4 chars per token).
    approx_chars = max(1, max_tokens * 4)
    if len(text) <= approx_chars:
        return text
    if not suffix:
        return text[:approx_chars]
    budget_chars = max(0, approx_chars - len(suffix))
    return text[:budget_chars] + suffix
