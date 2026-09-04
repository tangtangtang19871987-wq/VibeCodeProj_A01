"""03 — OpenCode as a LangGraph node.

LangGraph
   |
OpenCode
   |
LangGraph

The graph itself is trivial on purpose: one deterministic node builds a
bounded AgentTask, one node hands it to OpenCode via the shared adapter
(src/opencode_adapter.py), one deterministic node reports the outcome.
OpenCode's own internal loop — however many turns it actually takes — is
invisible to the graph; only AgentTask goes in and AgentResult comes out.
See docs/architecture.md for why that boundary is drawn exactly there.

This example doesn't verify anything yet (that's 05) and doesn't retry
(that's 06) — it exists purely to show the node shape.

Run: python examples/03_opencode_as_node/main.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph

from src.contracts import AgentResult, AgentTask
from src.opencode_adapter import OpenCodeAdapter


class State(TypedDict):
    task_id: str
    instruction: str
    workspace: str
    result_summary: str
    claimed_done: bool


def prepare_task(state: State) -> dict:
    """Deterministic: decides what OpenCode is asked to do, and where."""
    workspace = Path(state["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.txt").write_text("(empty)\n")
    return {}


def run_opencode(state: State) -> dict:
    """The entire boundary crossing: one AgentTask in, one AgentResult out."""
    task = AgentTask(
        task_id=state["task_id"],
        instruction=state["instruction"],
        workspace=Path(state["workspace"]),
        context={
            "simulate_attempt": lambda ws, attempt, last_error: (
                ws.joinpath("notes.txt").write_text("OpenCode was here.\n") or "wrote notes.txt"
            )
        },
        max_turns=2,
    )
    result: AgentResult = OpenCodeAdapter().run(task)
    return {"result_summary": result.summary, "claimed_done": result.claimed_done}


def report(state: State) -> dict:
    print(f"OpenCode claimed_done={state['claimed_done']}: {state['result_summary']}")
    return {}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("prepare_task", prepare_task)
    graph.add_node("run_opencode", run_opencode)
    graph.add_node("report", report)
    graph.add_edge(START, "prepare_task")
    graph.add_edge("prepare_task", "run_opencode")
    graph.add_edge("run_opencode", "report")
    graph.add_edge("report", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    workspace = tempfile.mkdtemp(prefix="opencode-node-")
    result = app.invoke(
        {
            "task_id": "03-demo",
            "instruction": "Write a short note into notes.txt describing what you did.",
            "workspace": workspace,
            "result_summary": "",
            "claimed_done": False,
        }
    )
    print("workspace:", workspace)
    print("notes.txt contents:", (Path(workspace) / "notes.txt").read_text().strip())
    assert result["claimed_done"] is True


if __name__ == "__main__":
    main()
