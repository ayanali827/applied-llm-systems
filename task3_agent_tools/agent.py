"""Assemble and run the Task 3 tool-using agent.

The agent is a LangChain create_agent graph: a chat model (our shared
fallback chain) plus the tools from tools.py. A run is a growing list of
messages: human question, AI tool calls, tool observations, and finally
an AI answer. run_agent() executes one question and reports both the
final answer and the tool calls the model chose, in order.
"""

from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage

from common.llm import extract_text, get_llm_with_fallbacks
from task3_agent_tools.tools import (
    calculator,
    list_data_files,
    read_text_file,
    web_search,
)

TOOLS = [calculator, list_data_files, read_text_file, web_search]

SYSTEM_PROMPT = """- For current or external facts, use web_search and cite the URLs you used. Search at most once per question; only search again if the first results clearly do not answer it. Rewording the same search is not a reason. Prefer answering from results you already have.
- For any arithmetic, use the calculator tool. Never do math in your head.
- For questions about the agent's files, use list_data_files and read_text_file.
- For current or external facts, use web_search and cite the URLs you used.
- If a tool returns an Error, explain the problem briefly instead of inventing a result.
- Keep final answers short and concrete."""

# Fuse against runaway loops: LangGraph counts each model/tool round
# toward this limit and stops the run when it is exceeded.
MAX_STEPS = 30


@dataclass
class AgentResult:
    """Outcome of one agent run."""

    question: str
    answer: str
    tool_calls: list[str] = field(default_factory=list)


def build_agent():
    """Create the agent graph with the shared fallback model chain."""
    return create_agent(
        model=get_llm_with_fallbacks(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            # Hard rails: enforced by code, whatever the model decides.
            ToolCallLimitMiddleware(tool_name="web_search", run_limit=2),
            ToolCallLimitMiddleware(run_limit=10),
        ],
    )

def _collect_tool_calls(messages: list) -> list[str]:
    """List the tool calls the model made, in order, as readable text."""
    calls: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                calls.append(f"{call['name']}({call['args']})")
    return calls


def run_agent(question: str, agent=None) -> AgentResult:
    """Run one question through the agent and return answer + tool calls."""
    text = question.strip()
    if not text:
        raise ValueError("Question is empty.")
    graph = agent if agent is not None else build_agent()
    try:
        result = graph.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"recursion_limit": MAX_STEPS},
        )
    except Exception as exc:  # step-limit trips and API failures land here
        return AgentResult(
            question=text,
            answer=f"Error: The agent run failed: {exc}",
            tool_calls=[],
        )
    messages = result["messages"]
    return AgentResult(
        question=text,
        answer=extract_text(messages[-1]).strip(),
        tool_calls=_collect_tool_calls(messages),
    )
