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
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_hint: Optional[str] = None) -> int:
    enc = _get_encoder(model_hint)
    return len(enc.encode(text or ""))


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
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Leave room for suffix tokens.
    suffix_tokens = enc.encode(suffix) if suffix else []
    budget = max(0, max_tokens - len(suffix_tokens))
    truncated = enc.decode(tokens[:budget])
    return truncated + (suffix if suffix else "")

