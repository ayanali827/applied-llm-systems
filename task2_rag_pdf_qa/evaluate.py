"""Evaluate the Task 2 QA system against questions.json.

The checks are deliberately simple and honest: for answerable
questions, the expected page must appear among the sources and the
answer must contain the expected words; for unanswerable ones, the
answer must be exactly the refusal sentence. This is not a full
semantic grading of answer quality.

Run from the project root (after ingesting a PDF):
    python -m task2_rag_pdf_qa.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from task2_rag_pdf_qa.qa import NOT_FOUND_MESSAGE, AnswerResult, answer_question
from task2_rag_pdf_qa.store import load_index

QUESTIONS_PATH: Path = Path(__file__).parent / "questions.json"


def load_cases(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Load the evaluation cases from the questions JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not hold a non-empty list.
    """
    if not path.is_file():
        raise FileNotFoundError(f"evaluation file not found: {path}")
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("questions file must contain a non-empty list")
    return data


def check_case(result: AnswerResult, case: dict[str, Any]) -> list[str]:
    """Check one QA result against its expected facts.

    Returns:
        A list of problem descriptions; empty means the case passed.
    """
    problems: list[str] = []
    answer: str = result.answer

    if case.get("expect_refusal"):
        if answer != NOT_FOUND_MESSAGE:
            problems.append("expected the refusal sentence, got an answer")
        return problems

    if answer == NOT_FOUND_MESSAGE:
        problems.append("system refused, but an answer was expected")
        return problems

    expected_page: Any = case.get("expected_page")
    if expected_page is not None:
        pages: set[int] = {
            int(doc.metadata.get("page_number", 0)) for doc in result.sources
        }
        if int(expected_page) not in pages:
            problems.append(
                f"expected a source from page {expected_page}, "
                f"got pages {sorted(pages)}"
            )

    lowered: str = answer.lower()
    for word in case.get("must_contain", []):
        if str(word).lower() not in lowered:
            problems.append(f"answer is missing expected word: {word!r}")
    return problems

def main() -> int:
    """Run all evaluation cases against the current index.

    Returns:
        Exit code: 0 if every case passes, 1 otherwise.
    """
    index = load_index()
    if index is None:
        print(
            "Error: no index found. Run the ingest command first.",
            file=sys.stderr,
        )
        return 1

    try:
        cases: list[dict[str, Any]] = load_cases()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    passed: int = 0
    for number, case in enumerate(cases, start=1):
        question: str = str(case["question"])
        answer_text: str = ""
        try:
            result: AnswerResult = answer_question(index, question)
            answer_text = result.answer
            problems: list[str] = check_case(result, case)
        except Exception as exc:  # one bad case must not kill the run
            problems = [f"asking failed: {type(exc).__name__}: {exc}"]
        if problems:
            print(f"[{number}/{len(cases)}] FAIL: {question}")
            for problem in problems:
                print(f"    - {problem}")
        else:
            passed += 1
            print(f"[{number}/{len(cases)}] PASS: {question}")
        if answer_text:
            print(f"    Answer: {answer_text}")

    print(f"\n{passed}/{len(cases)} checks passed.")
    return 0 if passed == len(cases) else 1



if __name__ == "__main__":
    raise SystemExit(main())
