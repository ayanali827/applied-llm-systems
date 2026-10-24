"""Offline tests for the Task 1 helper functions.

No LLM is called here: these check pure text functions and the prompt
builders only.
"""

from __future__ import annotations

import pytest

from task1_structured_extraction.extractor import _strip_code_fences
from task1_structured_extraction.prompts import (
    build_invoice_messages,
    build_resume_messages,
)


def test_strip_code_fences_removes_json_fences() -> None:
    """Markdown fences around JSON are removed."""
    raw: str = '```json\n{"a": 1}\n```'
    assert _strip_code_fences(raw) == '{"a": 1}'


def test_strip_code_fences_leaves_plain_text_unchanged() -> None:
    """Text without fences is returned as-is."""
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_build_invoice_messages_rejects_empty_text() -> None:
    """An empty document must fail fast, before any LLM call."""
    with pytest.raises(ValueError):
        build_invoice_messages("   ")


def test_build_resume_messages_shape_and_content() -> None:
    """The resume prompt has 4 messages, and the last holds our document."""
    messages = build_resume_messages("MY DOCUMENT")

    assert [m.type for m in messages] == ["system", "human", "ai", "human"]
    assert "MY DOCUMENT" in messages[-1].content
