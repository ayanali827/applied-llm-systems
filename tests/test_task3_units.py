"""Offline unit tests for Task 3 (tool agent).

Only pure/local logic is tested here: the calculator evaluator, the
sandbox path check, the search-result formatter, and helpers from
agent.py. No model or network calls.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from task3_agent_tools.agent import _collect_tool_calls, run_agent
from task3_agent_tools.tools import (
    MAX_SNIPPET_CHARS,
    calculate_expression,
    calculator,
    format_search_results,
    read_text_file,
    resolve_in_sandbox,
)

# --- calculator evaluator ---------------------------------------------------


def test_calculate_respects_precedence():
    assert calculate_expression("2 + 3 * 4") == 14.0


def test_calculate_parentheses_and_unary():
    assert calculate_expression("-(3 + 2) * 2") == -10.0


def test_calculate_division_and_modulo():
    assert calculate_expression("10 / 4") == 2.5
    assert calculate_expression("10 % 3") == 1.0


def test_calculate_rejects_names_and_calls():
    with pytest.raises(ValueError):
        calculate_expression("x + 1")
    with pytest.raises(ValueError):
        calculate_expression("abs(0 - 5)")


def test_calculate_rejects_empty_and_huge_power():
    with pytest.raises(ValueError):
        calculate_expression("   ")
    with pytest.raises(ValueError):
        calculate_expression("2 ** 1000")


def test_calculate_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        calculate_expression("1 / 0")


def test_calculator_tool_returns_strings():
    assert calculator.invoke({"expression": "2 + 3 * 4"}) == "14"
    assert calculator.invoke({"expression": "1 / 0"}).startswith("Error:")
    assert calculator.invoke({"expression": "abs(1)"}).startswith("Error:")


# --- sandbox -----------------------------------------------------------------


def test_resolve_in_sandbox_allows_inside(tmp_path):
    target = resolve_in_sandbox("notes/a.txt", root=tmp_path)
    assert target == (tmp_path / "notes" / "a.txt").resolve()


def test_resolve_in_sandbox_blocks_escape(tmp_path):
    with pytest.raises(ValueError):
        resolve_in_sandbox("../.env", root=tmp_path)
    with pytest.raises(ValueError):
        resolve_in_sandbox("sub/../../secret.txt", root=tmp_path)


def test_read_tool_blocks_env_escape():
    assert read_text_file.invoke({"path": "../.env"}).startswith("Error:")


def test_read_tool_reads_real_data_file():
    text = read_text_file.invoke({"path": "robot_specs.txt"})
    assert "Datheon Rover X2" in text


# --- search formatter ---------------------------------------------------------


def test_format_search_results_numbers_and_truncates():
    results = [
        {"title": "One", "url": "https://a.example", "content": "x" * 1000},
        {"title": "Two", "url": "https://b.example", "content": "short"},
    ]
    text = format_search_results(results)
    assert "1. One" in text
    assert "https://b.example" in text
    assert "x" * (MAX_SNIPPET_CHARS + 1) not in text


def test_format_search_results_empty():
    assert format_search_results([]) == "No results found."


# --- agent helpers -------------------------------------------------------------


def test_collect_tool_calls_in_order():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {"expression": "1 + 1"},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="The answer is 2."),
    ]
    assert _collect_tool_calls(messages) == ["calculator({'expression': '1 + 1'})"]


def test_run_agent_rejects_blank_question():
    with pytest.raises(ValueError):
        run_agent("   ")
