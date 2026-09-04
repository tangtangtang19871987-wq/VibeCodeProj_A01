"""08 — Disposable workspace.

create workspace -> hand to OpenCode -> verify -> promote or discard

LangGraph owns the whole lifecycle of the workspace OpenCode works in:
creating it, handing it over, and — after an independent verifier has
looked at the result — either promoting the useful output to a permanent
location or deleting the whole thing. OpenCode itself never decides any of
that; it just works inside whatever directory it's handed (see
docs/sandboxing.md: "prefer environment isolation over building a large
custom permission system").

Run twice in this example, with two different tasks against the same
graph, to show both outcomes: one OpenCode run actually fixes the bug (its
workspace gets promoted), one is asked to do something it fundamentally
can't do here (its workspace gets discarded, unpromoted, nothing leaks
out).

Run: python examples/08_disposable_workspace/main.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from langgraph.graph import END, START, StateGraph

from src.contracts import AgentTask
from src.opencode_adapter import OpenCodeAdapter
from src.verifier import run_pytest

FIXTURE = _HERE / "broken_project"
PROMOTED_DIR = _HERE / "promoted"


def _fix_correctly(workspace: Path, attempt: int, last_error: str) -> str:
    converter = workspace / "converter.py"
    converter.write_text(converter.read_text().replace("- 32", "+ 32"))
    return "fixed the sign error"


def _do_nothing(workspace: Path, attempt: int, last_error: str) -> str:
    return "unable to make progress on this task"


class State(TypedDict):
    workspace: str
    simulate_attempt_name: str
    verification_passed: bool
    outcome: str


def make_disposable_workspace(state: State) -> dict:
    workspace = tempfile.mkdtemp(prefix="disposable-")
    shutil.copytree(FIXTURE, workspace, dirs_exist_ok=True)
    return {"workspace": workspace}


def run_opencode(state: State) -> dict:
    simulate_attempt = _fix_correctly if state["simulate_attempt_name"] == "fix" else _do_nothing
    task = AgentTask(
        task_id=f"08-{state['simulate_attempt_name']}",
        instruction="Fix the bug so the tests pass.",
        workspace=Path(state["workspace"]),
        context={"simulate_attempt": simulate_attempt},
        max_turns=1,
    )
    OpenCodeAdapter().run(task)
    return {}


def verify(state: State) -> dict:
    verification = run_pytest(cwd=Path(state["workspace"]))
    return {"verification_passed": verification.passed}


def route(state: State) -> str:
    return "promote" if state["verification_passed"] else "discard"


def promote(state: State) -> dict:
    workspace = Path(state["workspace"])
    PROMOTED_DIR.mkdir(exist_ok=True)
    dest = PROMOTED_DIR / workspace.name
    shutil.copytree(workspace, dest, dirs_exist_ok=True)
    shutil.rmtree(workspace, ignore_errors=True)
    return {"outcome": f"PROMOTED to {dest}"}


def discard(state: State) -> dict:
    shutil.rmtree(state["workspace"], ignore_errors=True)
    return {"outcome": f"DISCARDED {state['workspace']} (nothing promoted)"}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("make_disposable_workspace", make_disposable_workspace)
    graph.add_node("run_opencode", run_opencode)
    graph.add_node("verify", verify)
    graph.add_node("promote", promote)
    graph.add_node("discard", discard)
    graph.add_edge(START, "make_disposable_workspace")
    graph.add_edge("make_disposable_workspace", "run_opencode")
    graph.add_edge("run_opencode", "verify")
    graph.add_conditional_edges("verify", route, {"promote": "promote", "discard": "discard"})
    graph.add_edge("promote", END)
    graph.add_edge("discard", END)
    return graph.compile()


def main() -> None:
    shutil.rmtree(PROMOTED_DIR, ignore_errors=True)
    app = build_graph()

    for name in ("fix", "give_up"):
        result = app.invoke(
            {
                "workspace": "",
                "simulate_attempt_name": name,
                "verification_passed": False,
                "outcome": "",
            }
        )
        print(f"[{name}] {result['outcome']}")

    promoted = sorted(p.name for p in PROMOTED_DIR.iterdir()) if PROMOTED_DIR.exists() else []
    print("promoted/ now contains:", promoted)
    assert len(promoted) == 1  # only the successful run's workspace survived


if __name__ == "__main__":
    main()
