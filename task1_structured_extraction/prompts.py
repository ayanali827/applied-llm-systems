"""Prompt construction for Task 1: invoice extraction.

The prompt is a ChatPromptTemplate: fixed instructions and one few-shot
example, with the target document (and machine-generated JSON schema)
supplied as variables at format time.
"""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from task1_structured_extraction.schemas import Invoice, Resume


EXAMPLE_INPUT: str = (
    "ACME SUPPLIES -- Invoice No INV-204 -- Date: 15 Jan 2026\n"
    "2 x Widget @ 50.00 = 100.00\n"
    "Subtotal: 100.00  Tax: 0.00  Total due: 100.00 USD"
)

# Money is strings here on purpose: "100.00" becomes an exact Decimal,
# while a JSON number risks binary-float representation error.
EXAMPLE_OUTPUT: dict = {
    "invoice_number": "INV-204",
    "vendor": "ACME SUPPLIES",
    "invoice_date": "2026-01-15",
    "currency": "USD",
    "line_items": [
        {"description": "Widget", "quantity": 2, "unit_price": "50.00", "line_total": "100.00"}
    ],
    "subtotal": "100.00",
    "tax": "0.00",
    "total": "100.00",
}

INVOICE_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise data-extraction engine. Extract the invoice "
            "from the user's document.\n"
            "Return ONLY one JSON object matching this schema:\n{schema_json}\n"
            "Rules:\n"
            "- Dates must be YYYY-MM-DD.\n"
            "- Amounts must be strings with two decimal places, e.g. \"50.00\".\n"
            "- Never invent values that are not in the document.\n"
            "- No explanations, no markdown, only the JSON object.",
        ),
        ("human", "Extract the invoice from this document:\n\n{example_input}"),
        ("ai", "{example_output}"),
        ("human", "Extract the invoice from this document:\n\n{document_text}"),
    ]
)


def build_invoice_messages(document_text: str) -> list[BaseMessage]:
    """Build the full message list for extracting one invoice.

    Args:
        document_text: The raw, unstructured invoice text.

    Returns:
        System, few-shot example, and target messages, ready for .invoke().

    Raises:
        ValueError: If document_text is empty or whitespace only.
    """
    if not document_text.strip():
        raise ValueError("document_text must not be empty.")

    return INVOICE_PROMPT.format_messages(
        schema_json=json.dumps(Invoice.model_json_schema(), indent=2),
        example_input=EXAMPLE_INPUT,
        example_output=json.dumps(EXAMPLE_OUTPUT, indent=2),
        document_text=document_text,
    )

RESUME_EXAMPLE_INPUT: str = (
    "Ali Raza -- ali.raza@example.com -- Skills: Python, SQL\n"
    "Data Analyst at FinCorp, Mar 2022 to Jun 2024\n"
    "BSc Computer Science, University of Karachi, 2021"
)

RESUME_EXAMPLE_OUTPUT: dict = {
    "full_name": "Ali Raza",
    "email": "ali.raza@example.com",
    "phone": None,
    "summary": None,
    "skills": ["Python", "SQL"],
    "experience": [
        {
            "company": "FinCorp",
            "title": "Data Analyst",
            "start_date": "2022-03-01",
            "end_date": "2024-06-30",
        }
    ],
    "education": [
        {
            "institution": "University of Karachi",
            "degree": "BSc Computer Science",
            "graduation_year": 2021,
        }
    ],
}

RESUME_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise data-extraction engine. Extract the resume "
            "from the user's document.\n"
            "Return ONLY one JSON object matching this schema:\n{schema_json}\n"
            "Rules:\n"
            "- Dates must be YYYY-MM-DD; use null for end_date if the job is current.\n"
            "- Never invent values that are not in the document; use null for optional missing fields.\n"
            "- No explanations, no markdown, only the JSON object.",
        ),
        ("human", "Extract the resume from this document:\n\n{example_input}"),
        ("ai", "{example_output}"),
        ("human", "Extract the resume from this document:\n\n{document_text}"),
    ]
)

def build_resume_messages(document_text: str) -> list[BaseMessage]:
    """Build the full message list for extracting one resume.

    Args:
        document_text: The raw, unstructured resume text.

    Returns:
        System, few-shot example, and target messages, ready for .invoke().

    Raises:
        ValueError: If document_text is empty or whitespace only.
    """
    if not document_text.strip():
        raise ValueError("document_text must not be empty.")

    return RESUME_PROMPT.format_messages(
        schema_json=json.dumps(Resume.model_json_schema(), indent=2),
        example_input=RESUME_EXAMPLE_INPUT,
        example_output=json.dumps(RESUME_EXAMPLE_OUTPUT, indent=2),
        document_text=document_text,
    )
