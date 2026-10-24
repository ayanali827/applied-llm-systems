"""Factory for the shared embedding model.

An embedding model turns text into a list of numbers (a vector).
Texts with similar meaning get vectors that are close together,
which is what makes similarity search possible.
"""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

from common.config import get_settings

DEFAULT_GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
DEFAULT_OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"


def get_embeddings() -> Embeddings:
    """Create the embedding model for the configured provider.

    Returns:
        A LangChain Embeddings object. Call .embed_query(text) for one
        text or .embed_documents(texts) for many.

    Raises:
        RuntimeError: If the selected provider's package is missing.
    """
    settings = get_settings()  # also loads .env and fails fast on a missing key

    if settings.provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=os.environ.get("GEMINI_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL),
        )

    # OpenAI branch: lazy import, same pattern as common/llm.py.
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "LLM_PROVIDER=openai needs the langchain-openai package, "
            "which is not installed."
        ) from exc
    return OpenAIEmbeddings(
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL),
    )
