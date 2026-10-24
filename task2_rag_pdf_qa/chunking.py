"""Split page Documents into smaller, overlapping chunks.

Smaller chunks make similarity search precise: each chunk is about one
topic. Overlap protects ideas that sit on a chunk border from being
cut in half.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200


def split_pages(
    pages: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split page Documents into overlapping chunks.

    Args:
        pages: Page Documents from load_pdf_pages(). Their metadata
            (source, page_number) is copied onto every chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between neighbouring chunks.
            Must be smaller than chunk_size.

    Returns:
        Chunk Documents in document order. Each chunk also gains a
        "start_index" metadata entry: where it starts in its page.

    Raises:
        ValueError: If pages is empty, or the sizes are invalid.
    """
    if not pages:
        raise ValueError("No pages to split.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(pages)
    