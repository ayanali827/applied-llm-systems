"""Tools for the Task 3 agent: safe calculator, file reader, web search.

Every tool takes plain arguments and returns a string. Tool functions
never raise for expected problems (bad input, missing files): they return
an "Error: ..." string instead, so the agent sees the problem as an
observation and can react to it.
"""
import os 
import ast
import operator
from typing import Union
from pathlib import Path

from langchain_core.tools import tool

Number = Union[int, float]

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Guards against absurd inputs like 9 ** 9 ** 9 freezing the machine.
MAX_POW_EXPONENT = 100


def _eval_node(node: ast.AST) -> float:
    """Evaluate one node of a parsed arithmetic tree (whitelist only)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numbers are allowed, got {node.value!r}.")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW_EXPONENT:
            raise ValueError(f"Exponent too large (max {MAX_POW_EXPONENT}).")
        return float(operation(left, right))
    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return float(operation(_eval_node(node.operand)))
    raise ValueError(f"Expression element not allowed: {type(node).__name__}")


def calculate_expression(expression: str) -> float:
    """Evaluate a basic arithmetic expression safely (no eval).

    Allows numbers, parentheses, and + - * / // % **. Names, function
    calls, and anything else raise ValueError.
    """
    text = expression.strip()
    if not text:
        raise ValueError("Expression is empty.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ValueError(f"Could not parse expression: {expression!r}") from None
    return _eval_node(tree)

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Use this for any math. Input must be numbers and the operators
    + - * / // % ** with parentheses only, e.g. "(1200 * 0.16) + 45".
    No variables, no words, no function calls.
    """
    try:
        value = calculate_expression(expression)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return str(value)

DATA_DIR = Path(__file__).parent / "data"

# Cap the returned text so one huge file cannot flood the model's context.
MAX_FILE_CHARS = 8000


def resolve_in_sandbox(relative_path: str, root: Path = DATA_DIR) -> Path:
    """Resolve a path inside the sandbox folder, refusing escapes.

    Raises ValueError if the fully resolved path is not inside root.
    """
    base = root.resolve()
    target = (base / relative_path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Path escapes the data folder: {relative_path!r}")
    return target


@tool
def list_data_files() -> str:
    """List the files available in the agent's data folder.

    Use this first when you need to know which files exist before
    reading one with read_text_file.
    """
    base = DATA_DIR.resolve()
    if not base.is_dir():
        return "Error: The data folder does not exist."
    files = sorted(
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file()
    )
    if not files:
        return "(the data folder is empty)"
    return "\n".join(files)


@tool
def read_text_file(path: str) -> str:
    """Read a text file from the agent's data folder and return its content.

    The path is relative to the data folder, e.g. "robot_specs.txt".
    Only files inside the data folder can be read; paths that try to
    leave it (for example "../.env") are refused. Long files are
    truncated at 8000 characters.
    """
    try:
        target = resolve_in_sandbox(path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not target.is_file():
        return f"Error: File not found in the data folder: {path!r}"
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: File is not readable text: {path!r}"
    except OSError as exc:
        return f"Error: Could not read the file: {exc}"
    if len(text) > MAX_FILE_CHARS:
        return text[:MAX_FILE_CHARS] + "\n[... truncated ...]"
    return text

MAX_SEARCH_RESULTS = 3
MAX_SNIPPET_CHARS = 300


def _make_tavily_client():
    """Build a Tavily search client, or raise ValueError if unconfigured."""
    from langchain_tavily import TavilySearch

    from common.config import get_settings

    get_settings()  # loads the .env file
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set in the .env file.")
    return TavilySearch(
    max_results=MAX_SEARCH_RESULTS,
    search_depth="basic",
    api_key=api_key,
)



def format_search_results(results: list) -> str:
    """Format raw Tavily result dicts as a numbered text block."""
    if not results:
        return "No results found."
    blocks = []
    for index, item in enumerate(results, start=1):
        title = str(item.get("title") or "Untitled").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or "").strip()[:MAX_SNIPPET_CHARS]
        blocks.append(f"{index}. {title}\n{url}\n{snippet}")
    return "\n\n".join(blocks)


@tool
def web_search(query: str) -> str:
    """Search the live web for current information and return top results.

    Use this for facts that may have changed or that you do not know:
    news, prices, versions, documentation. Returns numbered results,
    each with a title, a URL, and a short snippet. Cite the URLs you
    use in your final answer.
    """
    text = query.strip()
    if not text:
        return "Error: Search query is empty."
    try:
        client = _make_tavily_client()
        raw = client.invoke({"query": text})
    except Exception as exc:  # network/API problems become observations
        return f"Error: Web search failed: {exc}"
    results = raw.get("results", []) if isinstance(raw, dict) else []
    return format_search_results(results)
