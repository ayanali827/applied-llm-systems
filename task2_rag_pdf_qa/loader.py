"""Load a PDF file into one LangChain Document per page.

Each returned Document keeps metadata with the source file and a
1-based page number, which the QA step later uses for citations.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    """Load a PDF as a list of per-page Documents.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        One Document per page, in page order. Each Document's metadata
        gains a 1-based "page_number" entry for human-facing citations.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF, or no page has any
            extractable text (a scanned PDF has no text layer).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    pages: list[Document] = PyPDFLoader(str(pdf_path)).load()

    for page in pages:
        # PyPDFLoader counts pages from 0; humans count from 1.
        page.metadata["page_number"] = int(page.metadata.get("page", 0)) + 1

    if not any(page.page_content.strip() for page in pages):
        raise ValueError(
            f"No extractable text in {pdf_path}. "
            "The PDF is probably scanned images; OCR is out of scope."
        )
    return pages
    