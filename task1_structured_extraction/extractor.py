"""Extraction pipelines for Task 1.

One shared, generic retry loop serves every schema: call the LLM, parse
and validate the output, and on failure feed the exact error back to the
model for a bounded number of retries.
"""


from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from common.llm import extract_text, get_llm

from task1_structured_extraction.prompts import (
    build_invoice_messages,
    build_resume_messages,
)
from task1_structured_extraction.schemas import Invoice, Resume



class ExtractionError(RuntimeError):
    """Raised when extraction still fails after all retry attempts."""


SchemaT = TypeVar("SchemaT", bound=BaseModel)



def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences, if the model added any.

    Args:
        text: Raw model output, possibly wrapped in ```json ... ```.

    Returns:
        The text with the first/last fence lines removed.
    """
    stripped: str = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines: list[str] = stripped.splitlines()
    lines = lines[1:]  # drop the opening ``` or ```json line
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]  # drop the closing fence line
    return "\n".join(lines).strip()


def _extract_with_retries(
    document_text: str,
    schema: type[SchemaT],
    build_messages: Callable[[str], list[BaseMessage]],
    max_attempts: int = 3,
) -> SchemaT:
    """Extract a validated schema object from raw document text.

    Args:
        document_text: The unstructured document text.
        schema: The Pydantic model to validate the extracted data against.
        build_messages: A function to build the message list for the LLM.
        max_attempts: Maximum number of LLM calls before giving up.

    Returns:
        A validated schema object — never a raw dict or string.

    Raises:
        ValueError: If max_attempts is less than 1.
        ExtractionError: If every attempt fails parsing or validation.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    llm = get_llm()  # temperature defaults to 0.0: deterministic repairs
    messages: list[BaseMessage] = build_messages(document_text)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        raw: str = extract_text(llm.invoke(messages)).strip()

        try:
            data: object = json.loads(_strip_code_fences(raw))
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            # Feed back BOTH the bad output and the precise error, so the
            # model repairs that specific fault instead of starting over.
            messages = [
                *messages,
                AIMessage(content=raw),
                HumanMessage(
                    content=(
                        f"Your previous output failed validation:\n{exc}\n"
                        "Return ONLY a corrected JSON object matching the "
                        "schema. No explanations, no markdown."
                    )
                ),
            ]

    raise ExtractionError(
        f"Extraction failed after {max_attempts} attempts. Last error: {last_error}"
    ) from last_error

def extract_invoice(document_text: str, max_attempts: int = 3) -> Invoice:
    """Extract a validated Invoice from raw document text."""
    return _extract_with_retries(
        document_text, Invoice, build_invoice_messages, max_attempts
    )


def extract_resume(document_text: str, max_attempts: int = 3) -> Resume:
    """Extract a validated Resume from raw document text."""
    return _extract_with_retries(
        document_text, Resume, build_resume_messages, max_attempts
    )

