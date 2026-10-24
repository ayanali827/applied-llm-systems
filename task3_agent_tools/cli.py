"""Command-line interface for the Task 3 tool agent.

Usage:
    python -m task3_agent_tools.cli ask "your question"

Prints the tool calls the agent chose (transparency), then the answer.
"""

import argparse
import sys

from task3_agent_tools.agent import run_agent


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested command."""
    parser = argparse.ArgumentParser(
        prog="task3-agent",
        description="Ask the Task 3 tool-using agent a question.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask_parser = subparsers.add_parser("ask", help="Ask the agent a question.")
    ask_parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args(argv)

    if args.command == "ask":
        try:
            result = run_agent(args.question)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if result.tool_calls:
            print("Tool calls:")
            for call in result.tool_calls:
                print(f"- {call}")
            print()
        print(result.answer)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
