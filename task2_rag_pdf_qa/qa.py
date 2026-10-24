"""Question answering over the Task 2 FAISS index.

Rules enforced here:
- The model may use ONLY the retrieved chunks as its source of truth.
- If the chunks do not contain the answer, it must reply with one
  exact sentence (NOT_FOUND_MESSAGE) instead of guessing.
- Retrieved chunks are returned with the answer, so the caller can
  print citations (file name + page number).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS

from common.llm import extract_text, get_llm_with_fallbacks

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

NOT_FOUND_MESSAGE: str = "I don't know based on the provided documents."
DEFAULT_K: int = 4

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about documents.\n"
            "Rules:\n"
            "1. Use ONLY the context provided below. Do not use any "
            "outside knowledge.\n"
            "2. If the context does not contain the answer, reply with "
            f"exactly this sentence and nothing else: {NOT_FOUND_MESSAGE}\n"
            "3. Keep the answer short and factual.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


def format_context(docs: list[Document]) -> str:
    """Format retrieved chunks as one labelled context block.

    Args:
        docs: Retrieved chunk Documents.

    Returns:
        One string where each chunk is headed by its source file name
        and 1-based page number, e.g. "[Source: paper.pdf, page 2]".
    """
    blocks: list[str] = []
    for doc in docs:
        source_name: str = Path(str(doc.metadata.get("source", "unknown"))).name
        page_number: int = int(doc.metadata.get("page_number", 0))
        blocks.append(
            f"[Source: {source_name}, page {page_number}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def build_qa_messages(question: str, docs: list[Document]) -> list[BaseMessage]:
    """Build the QA prompt messages for one question.

    Args:
        question: The user's question. Must not be blank.
        docs: Retrieved chunks to use as the only source of truth.

    Returns:
        Messages ready for the chat model.

    Raises:
        ValueError: If the question is empty or whitespace only.
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")
    return QA_PROMPT.format_messages(
        context=format_context(docs),
        question=question.strip(),
    )

@dataclass(frozen=True)
class AnswerResult:
    """One QA result: the answer text plus the chunks behind it.

    Attributes:
        answer: The model's answer, or NOT_FOUND_MESSAGE.
        sources: The retrieved chunks the answer is based on. The
            caller turns these into citations (file + page).
    """

    answer: str
    sources: list[Document]


def _answer_once(
    index: FAISS,
    question: str,
    k: int = DEFAULT_K,
) -> AnswerResult:
    """Answer a question using only chunks retrieved from the index.

    Args:
        index: The loaded FAISS index.
        question: The user's question.
        k: How many chunks to retrieve as context.

    Returns:
        An AnswerResult with the answer text and the source chunks.

    Raises:
        ValueError: If k is less than 1, or the question is blank
            (raised by build_qa_messages).
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    docs: list[Document] = index.similarity_search(question, k=k)
    if not docs:
        return AnswerResult(answer=NOT_FOUND_MESSAGE, sources=[])

    response = get_llm_with_fallbacks().invoke(build_qa_messages(question, docs))
    return AnswerResult(answer=extract_text(response).strip(), sources=docs)



HISTORY_PATH: Path = Path(__file__).parent / "qa_history.jsonl"


def _record_answer(
    question: str,
    result: AnswerResult,
    path: Path = HISTORY_PATH,
) -> None:
    """Append one answered question to the JSONL history file.

    Each line is one JSON object: when it was asked, the question,
    the answer, and the deduplicated (file, page) sources used.
    """
    sources: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for doc in result.sources:
        name: str = Path(str(doc.metadata.get("source", "unknown"))).name
        page: int = int(doc.metadata.get("page_number", 0))
        if (name, page) not in seen:
            seen.add((name, page))
            sources.append({"source": name, "page": page})

    record: dict[str, object] = {
        "asked_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": result.answer,
        "sources": sources,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer_question(
    index: FAISS, question: str, k: int = DEFAULT_K
) -> AnswerResult:
    """Answer a question and record the result in the history file.

    Returns:
        The AnswerResult produced by the underlying QA call.
    """
    result: AnswerResult = _answer_once(index, question, k=k)
    _record_answer(question, result)
    return result
