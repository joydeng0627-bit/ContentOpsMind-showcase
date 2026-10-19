"""Helpers for normalizing Anthropic-compatible model responses."""

from __future__ import annotations

from typing import Any, Iterable, List


class EmptyModelResponseError(RuntimeError):
    """Raised when a model response contains no usable text blocks."""


def _content_blocks(response_or_blocks: Any) -> Iterable[Any]:
    if response_or_blocks is None:
        return []
    if isinstance(response_or_blocks, (list, tuple)):
        return response_or_blocks
    if isinstance(response_or_blocks, dict):
        return response_or_blocks.get("content") or []
    return getattr(response_or_blocks, "content", None) or []


def extract_model_text(response_or_blocks: Any) -> str:
    """Join non-empty text blocks from SDK objects or dictionary payloads."""
    parts: List[str] = []
    for block in _content_blocks(response_or_blocks):
        if isinstance(block, dict):
            block_type = block.get("type")
            text = block.get("text")
        else:
            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None)
        if block_type not in (None, "text"):
            continue
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    if not parts:
        raise EmptyModelResponseError("model response contained no usable text blocks")
    return "\n".join(parts)
