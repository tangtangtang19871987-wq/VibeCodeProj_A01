"""02 — Minimal agent loop: LLM -> tool -> observation -> LLM.

This is the smallest possible illustration of what an "agent harness" is:
something decides to call a tool, a tool runs and produces an observation,
that observation goes back to the decision-maker, repeat until done or out
of budget.

READ THIS TWICE: this is educational scaffolding, not a pattern to grow.
Real agent loops need to handle streaming, parallel tool calls, malformed
tool-call repair, context compression as history grows, sandboxing of what
a tool is allowed to touch, and dozens of other things that OpenCode (and
projects like it — see research/project_comparison.md) already solve well.
The instinct after writing this file is often "let's make this more
capable" — resist it. That instinct is exactly why example 03 exists:
instead of extending this loop, hand the task to OpenCode's own loop and
keep LangGraph out of its internals. This file should never grow past
what's needed to explain the concept.

The "LLM" here is a tiny deterministic stand-in, not a real model call —
see example 01 for how to wire in a real model; that mechanism is
intentionally not repeated here so this file stays about the loop shape,
not about LLM integration.

Run: python examples/02_minimal_agent_loop/main.py
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

MAX_TURNS = 4


class State(TypedDict):
    question: str
    turn: int
    observation: str | None
    final_answer: str | None
    transcript: list[str]
    pending_expression: str


def calculator_tool(expression: str) -> str:
    """The only tool available. Deliberately tiny and safe (no eval on
    arbitrary input — a whitelist of arithmetic characters only)."""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return "error: expression contains disallowed characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:
        return f"error: {exc}"


def agent_step(state: State) -> dict:
    """Stands in for an LLM deciding: call a tool, or finish.

    Turn 1: extract the arithmetic expression from the question and decide
    to call the calculator. Turn 2+: if an observation is available, use it
    to produce the final answer.
    """
    transcript = list(state["transcript"])
    if state["observation"] is None:
        match = re.search(r"[0-9+\-*/(). ]{3,}", state["question"])
        expression = match.group().strip() if match else ""
        transcript.append(f"agent: I should compute `{expression}` using the calculator tool.")
        return {"transcript": transcript, "pending_expression": expression}
    transcript.append(f"agent: the calculator returned {state['observation']}, so that's the answer.")
    return {"transcript": transcript, "final_answer": state["observation"]}


def tool_step(state: State) -> dict:
    expression = state.get("pending_expression", "")
    observation = calculator_tool(expression)
    transcript = [*state["transcript"], f"tool(calculator): {expression} -> {observation}"]
    return {"observation": observation, "transcript": transcript, "turn": state["turn"] + 1}


def route(state: State) -> str:
    if state.get("final_answer") is not None:
        return "done"
    if state["turn"] >= MAX_TURNS:
        return "give_up"
    return "call_tool"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("agent_step", agent_step)
    graph.add_node("tool_step", tool_step)
    graph.add_edge(START, "agent_step")
    graph.add_conditional_edges(
        "agent_step",
        route,
        {"call_tool": "tool_step", "done": END, "give_up": END},
    )
    graph.add_edge("tool_step", "agent_step")
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke(
        {
            "question": "What is 12 * (3 + 4)?",
            "turn": 0,
            "observation": None,
            "final_answer": None,
            "transcript": [],
            "pending_expression": "",
        }
    )
    for line in result["transcript"]:
        print(line)
    print("final answer:", result["final_answer"])
    assert result["final_answer"] == "84"


if __name__ == "__main__":
    main()
