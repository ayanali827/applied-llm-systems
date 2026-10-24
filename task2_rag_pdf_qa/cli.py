"""Command-line interface for Task 2: ingest PDFs, ask questions.

Usage:
    python -m task2_rag_pdf_qa.cli ingest <pdf-file>
    python -m task2_rag_pdf_qa.cli ask "<question>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from task2_rag_pdf_qa.chunking import split_pages
from task2_rag_pdf_qa.loader import load_pdf_pages
from task2_rag_pdf_qa.qa import DEFAULT_K, NOT_FOUND_MESSAGE, answer_question
from task2_rag_pdf_qa.store import ingest_chunks, load_index


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with ingest/ask subcommands.

    Returns:
        The configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="task2",
        description="Ask questions about PDF files, answered only from the PDFs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Add a PDF to the index.")
    ingest.add_argument("file", type=Path, help="Path to the PDF file.")

    ask = subparsers.add_parser("ask", help="Ask a question about the indexed PDFs.")
    ask.add_argument("question", help="The question to ask.")
    ask.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"How many chunks to retrieve (default: {DEFAULT_K}).",
    )
    return parser

def _run_ingest(pdf_file: Path) -> int:
    """Ingest one PDF: load, chunk, add to the index.

    Returns:
        Exit code: 0 on success, 1 on a user-facing error.
    """
    try:
        pages = load_pdf_pages(pdf_file)
        chunks = split_pages(pages)
        total: int = ingest_chunks(chunks)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Ingested {len(chunks)} chunks from {pdf_file.name}. "
        f"Index now holds {total} chunks."
    )
    return 0


def _run_ask(question: str, k: int) -> int:
    """Answer one question from the index, with citations.

    Returns:
        Exit code: 0 on success, 1 on a user-facing error.
    """
    index = load_index()
    if index is None:
        print(
            "Error: no index found. Run the ingest command first.",
            file=sys.stderr,
        )
        return 1

    try:
        result = answer_question(index, question, k=k)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result.answer)
    if result.answer != NOT_FOUND_MESSAGE and result.sources:
        print("\nSources:")
        seen: set[tuple[str, int]] = set()
        for doc in result.sources:
            name: str = Path(str(doc.metadata.get("source", "unknown"))).name
            page: int = int(doc.metadata.get("page_number", 0))
            if (name, page) not in seen:
                seen.add((name, page))
                print(f"- {name}, page {page}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the Task 2 CLI.

    Returns:
        Exit code from the chosen subcommand.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)
    if args.command == "ingest":
        return _run_ingest(args.file)
    return _run_ask(args.question, args.k)


if __name__ == "__main__":
    raise SystemExit(main())
