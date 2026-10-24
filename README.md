# applied-llm-systems

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-54%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellowgreen)

Four small, working LLM systems in Python with LangChain, plus the shared
plumbing they all stand on: validated structured extraction, retrieval
question-answering over PDFs, a tool-using agent, and a Streamlit chat UI
that combines the two middle systems.

Google Gemini is the default provider. A multi-provider fallback chain
(Gemini models, then Groq, then GitHub Models) keeps the systems answering
when one model is unavailable, and every answer path is covered by offline
tests and measured live evaluations.

## Contents

- [The four systems](#the-four-systems)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Design decisions worth noting](#design-decisions-worth-noting)
- [Evaluation and tests](#evaluation-and-tests)
- [Limitations](#limitations)
- [Layout](#layout)
- [License](#license)

## The four systems

### 1. Structured extraction (task1_structured_extraction/)
Turns messy invoice and resume text into validated Pydantic models.
The prompt's schema text is generated from the model itself
(model_json_schema()), and output passes three checks: JSON syntax,
field validation, and cross-field rules (line items must sum to the
subtotal; subtotal + tax must equal the total). Failures are retried
(max 3) with the exact validation error fed back to the model.

### 2. RAG over PDFs (task2_rag_pdf_qa/)
Question answering over PDF documents. PDFs are loaded page by page,
split into 1000-character chunks (200 overlap), embedded with
gemini-embedding-001, and stored in a FAISS index. Chunk IDs are
deterministic (file:p<page>:s<start>), so re-ingesting a file adds
nothing twice. Answers use only the top-4 retrieved chunks, cite the
file and page, and refuse with a fixed sentence when the documents do
not contain the answer. Every answer is also appended to a JSONL
history file.

### 3. Tool-using agent (task3_agent_tools/)
A LangChain create_agent agent with three safety-shaped tools:
a calculator that evaluates arithmetic through an AST whitelist
(never eval()), a file reader sandboxed to one data folder
(paths are resolved and checked before anything is read), and Tavily
web search. Guardrails come in two layers: a system-prompt rule
(soft) and ToolCallLimitMiddleware plus a recursion limit (hard).

### 4. Streamlit chat app (task4_streamlit_app/)
One chat UI with two modes - PDF questions (system 2) and Agent
(system 3) - a sidebar with PDF upload (ingested through the Task 2
pipeline) and a clear-chat button. Built around Streamlit's rerun
model: history in session_state, the index and agent in
cache_resource, answers in cache_data, both caches invalidated
after an upload-ingest.

## Architecture

    User
      |
      v
    Task 4 - Streamlit app
      |-- PDF mode   --> Task 2 - RAG over PDFs --> FAISS index (on disk)
      |-- Agent mode --> Task 3 - Tool-using agent --> Calculator / Files / Tavily

    Task 1, Task 2 and Task 3 all call:
      common/llm.py        - Gemini chat model + fallback chain
    Task 2 also uses:
      common/embeddings.py - gemini-embedding-001

## Setup

Requires Python 3.11+.

    git clone https://github.com/ayanali827/applied-llm-systems.git
    cd applied-llm-systems
    python -m venv .venv
    .venv\Scripts\activate        (Windows; Linux/macOS: source .venv/bin/activate)
    pip install -r requirements.txt
    copy .env.example .env         (Windows; Linux/macOS: cp .env.example .env)

Then fill in .env. GOOGLE_API_KEY is required for the default
provider. Optional: GROQ_API_KEY and GITHUB_TOKEN enable the
fallback groups, and TAVILY_API_KEY enables the agent's web search.

## Usage

    # Task 1 - extract an invoice or a resume
    python -m task1_structured_extraction.cli invoice task1_structured_extraction/samples/sample_invoice.txt
    python -m task1_structured_extraction.cli resume task1_structured_extraction/samples/sample_resume.txt

    # Task 2 - ingest a PDF, then ask questions
    python -m task2_rag_pdf_qa.make_sample_pdf   # optional: generate a sample PDF first
    python -m task2_rag_pdf_qa.cli ingest path/to/paper.pdf
    python -m task2_rag_pdf_qa.cli ask "How is robotics used in healthcare?"

    # Task 3 - ask the agent (it may calculate, read files, or search)
    python -m task3_agent_tools.cli ask "What is 18 * 24 + 5?"

    # Task 4 - the combined chat app
    streamlit run task4_streamlit_app/app.py

## Design decisions worth noting

- Fallbacks trigger on exceptions only. The RAG refusal sentence is
  a correct answer, not a failure; falling back on it could let a backup
  model invent ungrounded facts. Embeddings are never swapped for an
  existing index - a different embedding model means a different vector
  space, so the index would have to be rebuilt.
- FAISS replaced ChromaDB. ChromaDB's native code crashed with a
  Windows access violation (0xC0000005) inside add() on this machine,
  reproduced in isolation with two versions. FAISS plus deterministic
  chunk IDs gives the same persistence and idempotent ingest.
- Prompts ask, code forces. Behaviour that must hold (tool-call
  limits, sandbox, calculator whitelist) is enforced in code; the
  system prompt only guides.
- Store everything you draw. In the Streamlit app, a rerun redraws
  the conversation purely from session_state, so each stored message
  carries its text, its source labels, and its tool calls.

## Evaluation and tests

- 54 offline tests, all passing (python -m pytest): schemas and
  validators, loader/chunking/store units, calculator/sandbox/search
  units, evaluation checkers, and AppTest smoke tests for the UI.
- Task 2 live evaluation (6 questions over a research paper, run
  through the fallback chain while the primary model was quota-limited):
  5/6 strict. The one miss was a keyword check - the answer was verified
  correct by reading it; the check was kept strict rather than weakened.
- Task 3 live evaluation (5 cases through the real agent, served by a
  backup model): 3/5. One failure was the model garbling a correct
  calculator result while copying it into prose; the other was the
  model refusing a .env read before the tool call - the safe outcome,
  reached one layer earlier than the test expected.

## Limitations

- Free-tier quotas are real: when the primary model is limited, runs
  get slower and backup models answer slightly worse. Measured results
  above say which model served them.
- Retrieval ranking is approximate; k=4 and the evaluation exist
  because the top chunk is not always the right one.
- The RAG refusal is prompt-enforced, so it is measured by the
  evaluation rather than guaranteed.
- Keyword-based evaluation checks are proxies - they flag differences;
  a human still grades quality.
- Task 1: when a source document gives only a month and year, the
  model fills in the day (1st for starts, last day for ends). That
  precision is inferred, not sourced.
- The Streamlit app keeps one combined index for all uploaded PDFs;
  answers can mix sources across files.
- langchain-community is being sunset upstream; imports from it
  (PyPDFLoader, FAISS wrapper) raise a deprecation warning - migration
  to the standalone integration packages is future work.

## Layout

    common/                      config, LLM factory + fallback chain, embeddings
    task1_structured_extraction/ schemas, prompts, extractor, CLI, samples
    task2_rag_pdf_qa/            loader, chunking, FAISS store, QA, CLI, evaluation
    task3_agent_tools/           tools, agent, CLI, evaluation, data files
    task4_streamlit_app/         the combined Streamlit chat app
    tests/                       54 offline tests across all four tasks

## License

MIT — see [LICENSE](LICENSE).
