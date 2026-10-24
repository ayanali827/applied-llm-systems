"""Streamlit chat app for the internship project (Task 4).

Two modes, chosen in the sidebar:
- PDF questions: Task 2 RAG answers with a Sources expander.
- Agent: Task 3 agent answers with a Tool calls expander.

The sidebar also has a PDF upload (saved, then ingested with the
Task 2 pipeline) and a Clear chat button. Failures are shown as
short plain-language messages, never as raw tracebacks, and an
empty chat shows a hint with example questions.

Streamlit reruns this whole script on every interaction, so:
- the FAISS index and the agent are built once via st.cache_resource,
- RAG answers are cached by question text via st.cache_data,
- the chat history lives in st.session_state, and
- after an upload-ingest, both caches are cleared so the app sees
  the new index instead of the old one it is holding in memory.
"""

import sys
from pathlib import Path

# Streamlit only adds this file's own folder to the import path, so add
# the project root (one level up) to make the task packages importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from task2_rag_pdf_qa.chunking import split_pages
from task2_rag_pdf_qa.loader import load_pdf_pages
from task2_rag_pdf_qa.qa import NOT_FOUND_MESSAGE, AnswerResult, answer_question
from task2_rag_pdf_qa.store import ingest_chunks, load_index
from task3_agent_tools.agent import build_agent, run_agent

HISTORY_KEY = "messages"
UPLOADED_KEY = "uploaded_file_id"
DATA_DIR = PROJECT_ROOT / "task2_rag_pdf_qa" / "data"
ERROR_REPLY = "Sorry — something went wrong while answering. Please try again in a moment."


@st.cache_resource
def _get_index() -> FAISS | None:
    """Load the Task 2 FAISS index once and share it across reruns."""
    return load_index()


@st.cache_resource
def _get_agent():
    """Build the Task 3 agent once and share it across reruns."""
    return build_agent()


@st.cache_data(show_spinner=False)
def _cached_answer(_index: FAISS, question: str) -> AnswerResult:
    """Answer a question, remembering answers by question text.

    Repeat questions return the stored result instantly, with no model
    call. The leading underscore on _index tells Streamlit not to hash
    it (a FAISS index cannot be hashed); it is only passed through.
    """
    return answer_question(_index, question)


def _ensure_history() -> None:
    """Create the chat history list in session state if it is missing."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []


def _source_labels(sources: list[Document]) -> list[str]:
    """Build deduplicated 'file name, page N' labels from source chunks."""
    labels: list[str] = []
    seen: set[tuple[str, int]] = set()
    for doc in sources:
        name = Path(str(doc.metadata.get("source", "unknown file"))).name
        page = int(doc.metadata.get("page_number", 0))
        if (name, page) not in seen:
            seen.add((name, page))
            labels.append(f"{name}, page {page}")
    return labels


def _draw_message(message: dict) -> None:
    """Draw one stored chat message (bubble plus optional expanders)."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for label in message["sources"]:
                    st.markdown(f"- {label}")
        if message.get("tool_calls"):
            with st.expander("Tool calls"):
                for call in message["tool_calls"]:
                    st.markdown(f"- `{call}`")


def _draw_history() -> None:
    """Draw every saved message, oldest first."""
    for message in st.session_state[HISTORY_KEY]:
        _draw_message(message)


def _handle_upload() -> None:
    """Save and ingest an uploaded PDF once; refresh the caches after.

    The uploader returns the same file on every rerun, so the file's
    name+size is remembered in session state and each file is ingested
    only once. A file that cannot be ingested (for example a scanned
    PDF with no text) shows a plain error message instead of crashing,
    and is NOT marked as ingested, so it can be tried again.
    After a successful ingest, the cached index and cached answers
    are cleared: they still describe the old index.
    """
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded is None:
        return
    file_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get(UPLOADED_KEY) == file_id:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / uploaded.name
    target.write_bytes(uploaded.getvalue())
    try:
        with st.spinner(f"Ingesting {uploaded.name}..."):
            pages = load_pdf_pages(target)
            chunks = split_pages(pages)
            total = ingest_chunks(chunks)
    except Exception as exc:
        st.error(f"Could not ingest {uploaded.name}: {exc}")
        return
    st.session_state[UPLOADED_KEY] = file_id
    _get_index.clear()
    _cached_answer.clear()
    st.success(f"Ingested {uploaded.name}. The index now holds {total} chunks.")


def _draw_sidebar() -> str:
    """Draw the sidebar controls and return the selected mode."""
    with st.sidebar:
        st.header("Controls")
        mode = st.radio("Mode", ["PDF questions", "Agent"], horizontal=True)
        _handle_upload()
        if st.button("Clear chat"):
            st.session_state[HISTORY_KEY] = []
            st.rerun()
    return mode


def _ask_pdf(index: FAISS, question: str) -> dict:
    """Answer a question with the Task 2 RAG code (PDF mode)."""
    with st.spinner("Searching the PDF..."):
        result = _cached_answer(index, question)
    labels: list[str] = []
    if result.answer != NOT_FOUND_MESSAGE:
        labels = _source_labels(result.sources)
    return {
        "role": "assistant",
        "content": result.answer,
        "sources": labels,
        "tool_calls": [],
    }


def _ask_agent(question: str) -> dict:
    """Answer a question with the Task 3 agent (Agent mode)."""
    with st.status("Agent is working...", expanded=True) as status:
        result = run_agent(question, agent=_get_agent())
        for call in result.tool_calls:
            st.write(f"Tool call: {call}")
        status.update(label="Agent finished", state="complete")
    return {
        "role": "assistant",
        "content": result.answer,
        "sources": [],
        "tool_calls": result.tool_calls,
    }


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Chatbot")
    st.title("Chatbot")

    _ensure_history()
    mode = _draw_sidebar()
    _draw_history()

    if not st.session_state[HISTORY_KEY]:
        st.info(
            "Ask a question to start. PDF mode example: "
            "'How is robotics used in healthcare?' "
            "Agent mode example: 'What is 18 * 24 + 5?'"
        )

    index = _get_index()
    if mode == "PDF questions" and index is None:
        st.warning("No PDF index found yet. Ingest a PDF with the Task 2 CLI first.")
        return

    prompt = st.chat_input("Type your message")
    if prompt:
        user_message = {
            "role": "user",
            "content": prompt,
            "sources": [],
            "tool_calls": [],
        }
        st.session_state[HISTORY_KEY].append(user_message)
        _draw_message(user_message)

        try:
            if mode == "Agent":
                assistant_message = _ask_agent(prompt)
            else:
                assistant_message = _ask_pdf(index, prompt)
        except Exception:
            assistant_message = {
                "role": "assistant",
                "content": ERROR_REPLY,
                "sources": [],
                "tool_calls": [],
            }
        st.session_state[HISTORY_KEY].append(assistant_message)
        _draw_message(assistant_message)


main()
