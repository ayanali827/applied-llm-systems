"""Shared configuration for the Datheon LangChain internship project.

Loads settings from the environment (via a local .env file) and validates
them once, at startup, so every task fails fast on misconfiguration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Validated application settings. API keys are never shown in repr."""

    provider: str
    google_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)


def get_settings() -> Settings:
    """Load and validate settings from the environment.

    Returns:
        A validated Settings object.

    Raises:
        RuntimeError: If the provider is unknown, or the selected
            provider's API key is missing or empty.
    """
    load_dotenv()

    provider: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in {"gemini", "openai"}:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'gemini' or 'openai'.")

    # `or None` turns an empty string (blank in .env) into None = missing.
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None

    if provider == "gemini" and google_api_key is None:
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in.")
    if provider == "openai" and openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    return Settings(provider=provider, google_api_key=google_api_key, openai_api_key=openai_api_key)
