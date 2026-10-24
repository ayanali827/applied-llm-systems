"""Command-line interface for Task 1: structured extraction.

Usage:
    python -m task1_structured_extraction.cli invoice samples/sample_invoice.txt
    python -m task1_structured_extraction.cli resume samples/sample_resume.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from task1_structured_extraction.extractor import (
    ExtractionError,
    extract_invoice,
    extract_resume,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        A parser expecting a schema choice and an input file path.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Extract structured JSON from an invoice or resume text file."
    )
    parser.add_argument(
        "schema",
        choices=["invoice", "resume"],
        help="Which schema to extract.",
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the input text file.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of LLM calls before giving up (default: 3).",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Optional argument list for testing. None means: read the
            real command-line arguments.

    Returns:
        Exit code: 0 on success, 1 on any user-facing failure.
    """
    args: argparse.Namespace = build_parser().parse_args(argv)

    try:
        document_text: str = args.file.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        return 1

    try:
        if args.schema == "invoice":
            result = extract_invoice(document_text, max_attempts=args.max_attempts)
        else:
            result = extract_resume(document_text, max_attempts=args.max_attempts)
    except (ExtractionError, ValueError) as exc:
        # User-facing failures: short message on stderr, no traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # model_dump_json handles dates and Decimals correctly for us.
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
