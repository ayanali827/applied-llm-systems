"""Shared LLM factory for the Datheon LangChain internship project.

Every task obtains its chat model from get_llm(), so provider-specific
code exists in exactly one place.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage

import os

from langchain_core.language_models.chat_models import BaseChatModel

from common.config import get_settings


from typing import Any

from langchain_core.runnables import Runnable

DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"
DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
DEFAULT_GEMINI_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
)
DEFAULT_GROQ_FALLBACK_MODELS: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)
DEFAULT_GITHUB_FALLBACK_MODELS: tuple[str, ...] = ("openai/gpt-4o-mini",)
GITHUB_MODELS_BASE_URL: str = "https://models.github.ai/inference"



def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Create the chat model selected by LLM_PROVIDER in .env.

    Args:
        temperature: Sampling randomness. 0.0 for deterministic work
            (extraction, tests); higher values for creative tasks.

    Returns:
        A LangChain chat model with a common .invoke() interface.

    Raises:
        RuntimeError: If the selected provider is OpenAI but the
            optional langchain-openai package is not installed.
    """
    settings = get_settings()  # also loads .env, so keys are in os.environ

    if settings.provider == "gemini":
        # Imported here so an OpenAI-only user never needs this package.
        from langchain_google_genai import ChatGoogleGenerativeAI

        model: str = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        # No API key argument: the class reads GOOGLE_API_KEY from the env.
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    # OpenAI branch — lazy import, optional dependency.
    try:
        ChatOpenAI = __import__("langchain_openai", fromlist=["ChatOpenAI"]).ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LLM_PROVIDER=openai requires: pip install langchain-openai"
        ) from exc

    openai_model: str = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return ChatOpenAI(model=openai_model, temperature=temperature)

def extract_text(message: BaseMessage) -> str:
    """Extract plain text from a chat message.

    Content is a plain string for some providers, but a list of content
    blocks for others (e.g. Gemini 3). This normalises both to a string,
    keeping only text blocks and ignoring metadata such as signatures.

    Args:
        message: A LangChain chat message returned by .invoke().

    Returns:
        The concatenated text of all text blocks, or the string itself.
    """
    content: str | list = message.content

    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text: object = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)



def _model_names(env_name: str, defaults: tuple[str, ...]) -> list[str]:
    """Read a comma-separated model list from the environment.

    Args:
        env_name: Environment variable holding the list, if set.
        defaults: Model names to use when the variable is not set.

    Returns:
        The model names, in order, with blanks removed.
    """
    raw: str | None = os.getenv(env_name)
    if raw is None:
        return list(defaults)
    return [name.strip() for name in raw.split(",") if name.strip()]


def _gemini_chat(model: str, temperature: float) -> BaseChatModel:
    """Build one Gemini chat model by name."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def _groq_chat(model: str, temperature: float) -> BaseChatModel:
    """Build one Groq chat model by name (reads GROQ_API_KEY)."""
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=temperature)


def _github_chat(model: str, temperature: float) -> BaseChatModel:
    """Build one GitHub Models chat model by name (reads GITHUB_TOKEN)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=GITHUB_MODELS_BASE_URL,
        api_key=os.environ["GITHUB_TOKEN"],
    )


def get_llm_with_fallbacks(
    temperature: float = 0.0,
) -> Runnable[Any, BaseMessage]:
    """Create the primary chat model wrapped in a fallback chain.

    The primary comes from get_llm(). Backups are appended in a fixed
    order (other Gemini models, then Groq, then GitHub Models) and a
    backup group is included only when it is configured: Gemini
    fallbacks require the Gemini provider, Groq requires GROQ_API_KEY,
    GitHub Models requires GITHUB_TOKEN. On invoke(), the chain tries
    each model in order and returns the first successful response;
    it raises only if every model fails.

    Note: fallback triggers on exceptions (quota, server errors).
    A normal answer -- including the RAG refusal sentence -- is a
    success, not a failure, and never triggers the next model.

    Args:
        temperature: Sampling randomness, passed to every model.

    Returns:
        The primary model, or the primary wrapped with fallbacks.
    """
    settings = get_settings()  # also loads .env, so keys are in os.environ
    primary: BaseChatModel = get_llm(temperature)
    backups: list[BaseChatModel] = []

    if settings.provider == "gemini":
        primary_name: str = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        for name in _model_names(
            "GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACK_MODELS
        ):
            if name != primary_name:
                backups.append(_gemini_chat(name, temperature))

    if os.getenv("GROQ_API_KEY"):
        try:
            for name in _model_names(
                "GROQ_FALLBACK_MODELS", DEFAULT_GROQ_FALLBACK_MODELS
            ):
                backups.append(_groq_chat(name, temperature))
        except ImportError:
            pass  # Key set but package missing: skip this backup group.

    if os.getenv("GITHUB_TOKEN"):
        try:
            for name in _model_names(
                "GITHUB_FALLBACK_MODELS", DEFAULT_GITHUB_FALLBACK_MODELS
            ):
                backups.append(_github_chat(name, temperature))
        except ImportError:
            pass  # Key set but package missing: skip this backup group.

    if not backups:
        return primary
    return primary.with_fallbacks(backups)
