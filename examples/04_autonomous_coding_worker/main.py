"""04 — Autonomous coding worker.

A single bounded OpenCode task, pointed at a small broken project
(`broken_project/`: a Fahrenheit conversion function with a sign bug, and a
failing test suite). Inside that one task, OpenCode is free to inspect,
edit, run, and repeat as many times as `max_turns` allows — this is exactly
the "investigate, edit, run, debug, repair" cycle described in the project
brief, and exactly the part of the system this repository does not
reimplement (see fix_logic.py's docstring and
research/project_comparison.md).

Note what this example still does NOT do: it does not independently verify
the result (OpenCode's own `claimed_done` is trusted here, which is exactly
the gap example `05` exists to close).

Run: python examples/04_autonomous_coding_worker/main.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # repo root, for `src`
sys.path.insert(0, str(_HERE))  # this example's own dir, for fix_logic

from langgraph.graph import END, START, StateGraph

from fix_logic import simulate_attempt
from src.contracts import AgentTask
from src.opencode_adapter import OpenCodeAdapter

FIXTURE = _HERE / "broken_project"


class State(TypedDict):
    workspace: str
    summary: str
    claimed_done: bool
    turns_used: int


def make_workspace(state: State) -> dict:
    """Deterministic: a fresh, disposable copy of the broken project."""
    workspace = tempfile.mkdtemp(prefix="autonomous-worker-")
    shutil.copytree(FIXTURE, workspace, dirs_exist_ok=True)
    return {"workspace": workspace}


def run_opencode_worker(state: State) -> dict:
    workspace = Path(state["workspace"])
    task = AgentTask(
        task_id="04-fix-converter",
        instruction="The tests in this project are failing. Investigate, fix the bug, and re-run the tests until they pass.",
        workspace=workspace,
        context={
            "simulate_attempt": simulate_attempt,
            "check_command": [sys.executable, "-m", "pytest", "-q"],
        },
        max_turns=4,
    )
    result = OpenCodeAdapter().run(task)
    return {
        "summary": result.summary,
        "claimed_done": result.claimed_done,
        "turns_used": result.turns_used,
    }


def report(state: State) -> dict:
    print(f"OpenCode {'finished' if state['claimed_done'] else 'gave up'} "
          f"in {state['turns_used']} turn(s): {state['summary']}")
    return {}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("make_workspace", make_workspace)
    graph.add_node("run_opencode_worker", run_opencode_worker)
    graph.add_node("report", report)
    graph.add_edge(START, "make_workspace")
    graph.add_edge("make_workspace", "run_opencode_worker")
    graph.add_edge("run_opencode_worker", "report")
    graph.add_edge("report", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke({"workspace": "", "summary": "", "claimed_done": False, "turns_used": 0})
    log_path = Path(result["workspace"]) / "opencode_session.log"
    print(f"\nfull session log (never entered graph state — see docs/context_management.md):\n{log_path}")
    print(log_path.read_text())
    assert result["claimed_done"] is True
    assert result["turns_used"] == 2  # wrong fix, then the correct one


if __name__ == "__main__":
    main()
