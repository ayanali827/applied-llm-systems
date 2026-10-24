"""Offline tests for the Task 1 Pydantic schemas.

These tests call no LLM and need no API key: they check that the schemas
accept good data and reject bad data, exactly as designed.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from task1_structured_extraction.schemas import Invoice, LineItem, Resume


def valid_invoice_data() -> dict:
    """Return one known-good invoice as a plain dictionary.

    Returns:
        A dictionary that must always pass Invoice validation. Tests
        copy it and break one thing at a time.
    """
    return {
        "invoice_number": "INV-1",
        "vendor": "Acme",
        "invoice_date": "2026-01-15",
        "currency": "USD",
        "line_items": [
            {
                "description": "Widget",
                "quantity": 2,
                "unit_price": "50.00",
                "line_total": "100.00",
            }
        ],
        "subtotal": "100.00",
        "tax": "0.00",
        "total": "100.00",
    }


def test_valid_invoice_passes_and_coerces_types() -> None:
    """A good invoice passes, and strings become real date/Decimal objects."""
    invoice: Invoice = Invoice.model_validate(valid_invoice_data())

    assert invoice.invoice_date == date(2026, 1, 15)
    assert isinstance(invoice.invoice_date, date)
    assert invoice.line_items[0].unit_price == Decimal("50.00")
    assert isinstance(invoice.line_items[0].unit_price, Decimal)


def test_negative_quantity_is_rejected() -> None:
    """Quantity -3 is legal JSON but illegal in the real world."""
    with pytest.raises(ValidationError):
        LineItem(
            description="Widget",
            quantity=-3,
            unit_price="50.00",
            line_total="-150.00",
        )


def test_non_numeric_price_is_rejected() -> None:
    """The word 'fifty' cannot become a Decimal."""
    with pytest.raises(ValidationError):
        LineItem(
            description="Widget",
            quantity=1,
            unit_price="fifty",
            line_total="50.00",
        )

def test_line_items_must_sum_to_subtotal() -> None:
    """All fields can be legal, yet the numbers can still disagree."""
    data: dict = valid_invoice_data()
    data["subtotal"] = "120.00"
    data["total"] = "120.00"

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_subtotal_plus_tax_must_equal_total() -> None:
    """Subtotal + tax must equal the stated total."""
    data: dict = valid_invoice_data()
    data["tax"] = "10.00"  # total stays 100.00, but 100 + 10 = 110

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_one_cent_difference_is_allowed() -> None:
    """A difference of exactly 0.01 is inside the rounding tolerance."""
    data: dict = valid_invoice_data()
    data["subtotal"] = "100.01"
    data["total"] = "100.01"

    invoice: Invoice = Invoice.model_validate(data)
    assert invoice.subtotal == Decimal("100.01")


def test_line_total_must_match_quantity_times_price() -> None:
    """2 x 50.00 is 100.00, not 90.00."""
    with pytest.raises(ValidationError):
        LineItem(
            description="Widget",
            quantity=2,
            unit_price="50.00",
            line_total="90.00",
        )


def test_resume_end_date_before_start_date_is_rejected() -> None:
    """A job cannot end before it starts."""
    with pytest.raises(ValidationError):
        Resume.model_validate(
            {
                "full_name": "Test Person",
                "email": "test@example.com",
                "skills": ["Python"],
                "experience": [
                    {
                        "company": "Acme",
                        "title": "Dev",
                        "start_date": "2024-01-01",
                        "end_date": "2023-01-01",
                    }
                ],
                "education": [],
            }
        )


def test_bad_currency_code_is_rejected() -> None:
    """Currency must be 3 upper-case letters, e.g. USD — not 'usd'."""
    data: dict = valid_invoice_data()
    data["currency"] = "usd"

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)

def test_line_items_must_sum_to_subtotal() -> None:
    """All fields can be legal, yet the numbers can still disagree."""
    data: dict = valid_invoice_data()
    data["subtotal"] = "120.00"
    data["total"] = "120.00"

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_subtotal_plus_tax_must_equal_total() -> None:
    """Subtotal + tax must equal the stated total."""
    data: dict = valid_invoice_data()
    data["tax"] = "10.00"  # total stays 100.00, but 100 + 10 = 110

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_one_cent_difference_is_allowed() -> None:
    """A difference of exactly 0.01 is inside the rounding tolerance."""
    data: dict = valid_invoice_data()
    data["subtotal"] = "100.01"
    data["total"] = "100.01"

    invoice: Invoice = Invoice.model_validate(data)
    assert invoice.subtotal == Decimal("100.01")


def test_line_total_must_match_quantity_times_price() -> None:
    """2 x 50.00 is 100.00, not 90.00."""
    with pytest.raises(ValidationError):
        LineItem(
            description="Widget",
            quantity=2,
            unit_price="50.00",
            line_total="90.00",
        )


def test_resume_end_date_before_start_date_is_rejected() -> None:
    """A job cannot end before it starts."""
    with pytest.raises(ValidationError):
        Resume.model_validate(
            {
                "full_name": "Test Person",
                "email": "test@example.com",
                "skills": ["Python"],
                "experience": [
                    {
                        "company": "Acme",
                        "title": "Dev",
                        "start_date": "2024-01-01",
                        "end_date": "2023-01-01",
                    }
                ],
                "education": [],
            }
        )


def test_bad_currency_code_is_rejected() -> None:
    """Currency must be 3 upper-case letters, e.g. USD — not 'usd'."""
    data: dict = valid_invoice_data()
    data["currency"] = "usd"

    with pytest.raises(ValidationError):
        Invoice.model_validate(data)
