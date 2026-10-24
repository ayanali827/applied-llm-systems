"""Persistent FAISS vector store for Task 2 chunks.

Design note: this project originally used Chroma. On the development
machine, chromadb 1.x (tested 1.5.9 and 1.3.7) crashed with a Windows
access violation (exit code -1073741819) inside its native add() call,
even in-memory with ready-made embeddings. The store was therefore
switched to FAISS; embeddings, chunking and retrieval are unchanged.

FAISS is a file-based index, not a database server: the whole index is
saved to a folder and loaded back on demand. Idempotent ingest is done
explicitly: chunks carry deterministic IDs, and ingest skips any chunk
whose ID is already in the index.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from common.embeddings import get_embeddings

DEFAULT_INDEX_DIR: Path = Path(__file__).parent / "faiss_index"


def chunk_id(chunk: Document) -> str:
    """Build the deterministic ID for one chunk.

    Args:
        chunk: A chunk Document with source, page_number and
            start_index metadata.

    Returns:
        An ID like "paper.pdf:p2:s1450" — page 2, starting at
        character 1450 of that page. Same chunk, same ID, every run.
    """
    source_name: str = Path(str(chunk.metadata.get("source", "unknown"))).name
    page_number: int = int(chunk.metadata.get("page_number", 0))
    start_index: int = int(chunk.metadata.get("start_index", 0))
    return f"{source_name}:p{page_number}:s{start_index}"


def load_index(index_dir: Path = DEFAULT_INDEX_DIR) -> FAISS | None:
    """Load a saved FAISS index from disk.

    Args:
        index_dir: Folder previously written by save_index().

    Returns:
        The loaded index, or None if no index has been saved yet.
    """
    if not (index_dir / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        # The index file is written by this program (pickle format).
        # Only ever load index files you created yourself.
        allow_dangerous_deserialization=True,
    )


def save_index(index: FAISS, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
    """Save a FAISS index to disk, replacing any previous save."""
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_dir))


def count_chunks(index: FAISS) -> int:
    """Return how many chunks the index currently holds."""
    return int(index.index.ntotal)


def ingest_chunks(
    chunks: list[Document],
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> int:
    """Add new chunks to the persistent index, skipping ones already in it.

    Args:
        chunks: Chunk Documents from split_pages().
        index_dir: Where the index lives on disk.

    Returns:
        The total number of chunks in the index after ingest. Running
        ingest twice on the same PDF returns the same total both
        times: re-ingest adds nothing.

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError("No chunks to ingest.")

    index: FAISS | None = load_index(index_dir)
    existing_ids: set[str] = (
        set(index.index_to_docstore_id.values()) if index is not None else set()
    )
    new_chunks: list[Document] = [
        chunk for chunk in chunks if chunk_id(chunk) not in existing_ids
    ]

    if index is None:
        index = FAISS.from_documents(
            new_chunks,
            get_embeddings(),
            ids=[chunk_id(chunk) for chunk in new_chunks],
        )
    elif new_chunks:
        index.add_documents(
            new_chunks,
            ids=[chunk_id(chunk) for chunk in new_chunks],
        )

    save_index(index, index_dir)
    return count_chunks(index)
