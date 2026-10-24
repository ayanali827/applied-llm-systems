"""Pydantic schemas for Task 1: structured extraction.

These models are the contract that LLM output must satisfy before any
downstream code is allowed to trust it.
"""

from datetime import date
from decimal import Decimal
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, Field, model_validator


class LineItem(BaseModel):
    """One row of an invoice."""

    description: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    line_total: Decimal

    @model_validator(mode="after")
    def check_line_total(self) -> Self:
        """Check quantity x unit_price matches the stated line total."""
        expected: Decimal = self.unit_price * self.quantity
        if abs(expected - self.line_total) > Decimal("0.01"):
            raise ValueError(
                f"Line total mismatch: {self.quantity} x {self.unit_price} "
                f"= {expected}, but line_total is {self.line_total}."
            )
        return self

class Invoice(BaseModel):
    """A complete invoice extracted from unstructured text."""

    @model_validator(mode="after")
    def check_totals(self) -> Self:
        """Check line items sum to subtotal, and subtotal + tax = total."""
        tolerance: Decimal = Decimal("0.01")

        lines_sum: Decimal = sum(
            (item.line_total for item in self.line_items), Decimal("0")
        )
        if abs(lines_sum - self.subtotal) > tolerance:
            raise ValueError(
                f"Line items sum to {lines_sum}, but subtotal is {self.subtotal}."
            )

        if abs(self.subtotal + self.tax - self.total) > tolerance:
            raise ValueError(
                f"Subtotal {self.subtotal} + tax {self.tax} != total {self.total}."
            )
        return self

    invoice_number: str = Field(min_length=1)
    vendor: str = Field(min_length=1)
    invoice_date: date
    currency: str = Field(pattern=r"^[A-Z]{3}$")  # ISO code, e.g. USD, PKR, EUR
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal = Field(ge=0)
    tax: Decimal = Field(ge=0)
    total: Decimal = Field(ge=0)
    
class WorkExperience(BaseModel):
    """One job entry on a resume."""

    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_date: date
    end_date: date | None = None  # None means "current job"

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        """Check the end date is not before the start date."""
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date.")
        return self


class Education(BaseModel):
    """One education entry on a resume."""

    institution: str = Field(min_length=1)
    degree: str = Field(min_length=1)
    graduation_year: int = Field(ge=1950, le=2100)


class Resume(BaseModel):
    """A complete resume extracted from unstructured text."""

    full_name: str = Field(min_length=1)
    # Regex instead of EmailStr: EmailStr needs an extra package; this
    # pattern is weaker (no full RFC check) — a deliberate trade-off.
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone: str | None = None
    summary: str | None = None
    skills: list[str] = Field(min_length=1)
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
