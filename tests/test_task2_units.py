"""Offline unit tests for Task 2 (no network, no API key needed)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from reportlab.pdfgen import canvas

from task2_rag_pdf_qa.chunking import split_pages
from task2_rag_pdf_qa.loader import load_pdf_pages
from task2_rag_pdf_qa.store import chunk_id

import json

from common.llm import _model_names
from task2_rag_pdf_qa.evaluate import check_case, load_cases
from task2_rag_pdf_qa.qa import (
    NOT_FOUND_MESSAGE,
    AnswerResult,
    _record_answer,
    build_qa_messages,
    format_context,
)


def _make_pdf(path: Path, lines: list[str]) -> Path:
    """Write a small real PDF with the given text lines."""
    pdf: canvas.Canvas = canvas.Canvas(str(path))
    y: int = 800
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 20
    pdf.save()
    return path


def test_loader_missing_file(tmp_path: Path) -> None:
    """A non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pdf_pages(tmp_path / "nope.pdf")


def test_loader_rejects_non_pdf(tmp_path: Path) -> None:
    """A .txt file raises ValueError."""
    file_path: Path = tmp_path / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pdf_pages(file_path)


def test_loader_reads_generated_pdf(tmp_path: Path) -> None:
    """A generated one-page PDF loads with 1-based page_number."""
    pdf_path: Path = _make_pdf(tmp_path / "sample.pdf", ["Hello robotics world"])
    pages: list[Document] = load_pdf_pages(pdf_path)
    assert len(pages) == 1
    assert pages[0].metadata["page_number"] == 1
    assert "Hello robotics" in pages[0].page_content


def test_loader_rejects_blank_pdf(tmp_path: Path) -> None:
    """A PDF with no extractable text raises ValueError."""
    pdf_path: Path = _make_pdf(tmp_path / "blank.pdf", [])
    with pytest.raises(ValueError):
        load_pdf_pages(pdf_path)


def test_split_pages_rejects_empty_list() -> None:
    """Splitting zero pages raises ValueError."""
    with pytest.raises(ValueError):
        split_pages([])


def test_split_pages_rejects_overlap_not_smaller_than_size() -> None:
    """Overlap >= chunk_size raises ValueError."""
    doc: Document = Document(
        page_content="word " * 50,
        metadata={"source": "x.pdf", "page_number": 1},
    )
    with pytest.raises(ValueError):
        split_pages([doc], chunk_size=100, chunk_overlap=100)


def test_split_pages_keeps_metadata_and_size_limit() -> None:
    """Chunks keep the page metadata and stay within chunk_size."""
    doc: Document = Document(
        page_content="word " * 300,
        metadata={"source": "x.pdf", "page_number": 3},
    )
    chunks: list[Document] = split_pages([doc], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["page_number"] == 3 for c in chunks)


def test_chunk_id_is_deterministic() -> None:
    """Same chunk data always gives the same ID; a new start differs."""
    chunk: Document = Document(
        page_content="abc",
        metadata={"source": "paper.pdf", "page_number": 2, "start_index": 40},
    )
    same: Document = Document(
        page_content="abc",
        metadata={"source": "paper.pdf", "page_number": 2, "start_index": 40},
    )
    other: Document = Document(
        page_content="abc",
        metadata={"source": "paper.pdf", "page_number": 2, "start_index": 41},
    )
    assert chunk_id(chunk) == chunk_id(same) == "paper.pdf:p2:s40"
    assert chunk_id(chunk) != chunk_id(other)



def _doc(text: str, page: int) -> Document:
    """Build a fake chunk Document for QA/evaluate tests."""
    return Document(
        page_content=text,
        metadata={"source": "paper.pdf", "page_number": page},
    )


def test_format_context_labels_source_and_page() -> None:
    """Context text carries the [Source: file, page N] label."""
    text: str = format_context([_doc("Robots help doctors.", 2)])
    assert "[Source: paper.pdf, page 2]" in text
    assert "Robots help doctors." in text


def test_build_qa_messages_rejects_blank_question() -> None:
    """A blank question raises ValueError."""
    with pytest.raises(ValueError):
        build_qa_messages("   ", [_doc("text", 1)])


def test_build_qa_messages_shape_and_refusal_rule() -> None:
    """Messages are system + human; the system text holds the refusal."""
    messages = build_qa_messages("What is a sensor?", [_doc("A sensor sees.", 3)])
    assert len(messages) == 2
    assert NOT_FOUND_MESSAGE in str(messages[0].content)
    assert "What is a sensor?" in str(messages[1].content)


def test_check_case_refusal_expected_and_given() -> None:
    """Refusal case passes when the answer is exactly the refusal."""
    result: AnswerResult = AnswerResult(answer=NOT_FOUND_MESSAGE, sources=[])
    case: dict = {"expect_refusal": True}
    assert check_case(result, case) == []


def test_check_case_refusal_expected_but_answered() -> None:
    """Refusal case fails when the system answers anyway."""
    result: AnswerResult = AnswerResult(answer="Paris.", sources=[])
    case: dict = {"expect_refusal": True}
    assert len(check_case(result, case)) == 1


def test_check_case_answer_expected_but_refused() -> None:
    """Answerable case fails when the system refuses."""
    result: AnswerResult = AnswerResult(answer=NOT_FOUND_MESSAGE, sources=[])
    case: dict = {"expect_refusal": False, "expected_page": 2, "must_contain": []}
    assert len(check_case(result, case)) == 1


def test_check_case_wrong_page_is_a_problem() -> None:
    """A missing expected page is reported."""
    result: AnswerResult = AnswerResult(
        answer="Drones watch the enemy.", sources=[_doc("x", 2)]
    )
    case: dict = {
        "expect_refusal": False,
        "expected_page": 3,
        "must_contain": ["Drones"],
    }
    problems: list[str] = check_case(result, case)
    assert len(problems) == 1
    assert "page 3" in problems[0]


def test_check_case_full_pass() -> None:
    """Right page + required word present means no problems."""
    result: AnswerResult = AnswerResult(
        answer="The da Vinci robot helps surgeons.", sources=[_doc("x", 2)]
    )
    case: dict = {
        "expect_refusal": False,
        "expected_page": 2,
        "must_contain": ["da Vinci"],
    }
    assert check_case(result, case) == []


def test_check_case_must_contain_is_case_insensitive() -> None:
    """Word checks ignore letter case on both sides."""
    result: AnswerResult = AnswerResult(
        answer="SURVEILLANCE drones.", sources=[_doc("x", 3)]
    )
    case: dict = {
        "expect_refusal": False,
        "expected_page": 3,
        "must_contain": ["surveillance"],
    }
    assert check_case(result, case) == []


def test_load_cases_missing_file(tmp_path: Path) -> None:
    """A missing questions file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "nope.json")


def test_load_cases_rejects_non_list(tmp_path: Path) -> None:
    """A JSON object instead of a list raises ValueError."""
    path: Path = tmp_path / "bad.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_load_cases_reads_valid_file(tmp_path: Path) -> None:
    """A valid questions file loads into a list of cases."""
    path: Path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"question": "Q?"}]), encoding="utf-8")
    cases: list = load_cases(path)
    assert len(cases) == 1
    assert cases[0]["question"] == "Q?"


def test_model_names_defaults_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env variable, the default list is used."""
    monkeypatch.delenv("MY_MODELS", raising=False)
    assert _model_names("MY_MODELS", ("a", "b")) == ["a", "b"]


def test_model_names_splits_strips_and_drops_blanks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env lists are comma-split, trimmed, and blanks are dropped."""
    monkeypatch.setenv("MY_MODELS", " x, y ,, z ")
    assert _model_names("MY_MODELS", ("a",)) == ["x", "y", "z"]



def test_record_answer_appends_jsonl(tmp_path: Path) -> None:
    """Answers are appended as JSON lines, sources deduplicated."""
    path: Path = tmp_path / "history.jsonl"
    result: AnswerResult = AnswerResult(
        answer="Drones watch.",
        sources=[_doc("x", 3), _doc("y", 3)],
    )
    _record_answer("How are robots used in the military?", result, path=path)
    _record_answer(
        "What is the capital of France?",
        AnswerResult(answer=NOT_FOUND_MESSAGE, sources=[]),
        path=path,
    )

    lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first: dict = json.loads(lines[0])
    assert first["question"] == "How are robots used in the military?"
    assert first["answer"] == "Drones watch."
    assert first["sources"] == [{"source": "paper.pdf", "page": 3}]
    assert "asked_at" in first

    second: dict = json.loads(lines[1])
    assert second["answer"] == NOT_FOUND_MESSAGE
    assert second["sources"] == []
