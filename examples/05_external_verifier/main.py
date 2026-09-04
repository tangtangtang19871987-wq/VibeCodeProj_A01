"""05 — External deterministic verification.

OpenCode
   |
verifier
 /    \\
PASS  FAIL

OpenCode (via the fake backend's stand-in fix, see fix_logic.py) applies a
fix and reports `claimed_done=True` without checking its own work. The
graph does not take that claim at face value: `verify_result` runs the real
test suite with src/verifier.py, and a conditional edge branches on
*that* result, not on `claimed_done`. This example's fixture is set up so
those two disagree — OpenCode's fix looks plausible and is still wrong —
specifically to make the point visible: run this and watch the graph
correctly report FAIL even though OpenCode was confident.

No retry yet (that's 06). Both branches are terminal here.

Run: python examples/05_external_verifier/main.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE))

from langgraph.graph import END, START, StateGraph

from fix_logic import simulate_attempt
from src.contracts import AgentTask
from src.opencode_adapter import OpenCodeAdapter
from src.verifier import run_pytest

FIXTURE = _HERE / "broken_project"


class State(TypedDict):
    workspace: str
    claimed_done: bool
    opencode_summary: str
    verification_passed: bool
    verification_reason: str


def make_workspace(state: State) -> dict:
    workspace = tempfile.mkdtemp(prefix="external-verifier-")
    shutil.copytree(FIXTURE, workspace, dirs_exist_ok=True)
    return {"workspace": workspace}


def run_opencode(state: State) -> dict:
    task = AgentTask(
        task_id="05-fix-converter",
        instruction="The tests in this project are failing. Fix the bug.",
        workspace=Path(state["workspace"]),
        context={"simulate_attempt": simulate_attempt},  # no check_command: OpenCode won't self-verify
        max_turns=1,
    )
    result = OpenCodeAdapter().run(task)
    return {"claimed_done": result.claimed_done, "opencode_summary": result.summary}


def verify_result(state: State) -> dict:
    """The independent check. Never trusts claimed_done (docs/verification.md)."""
    verification = run_pytest(cwd=Path(state["workspace"]))
    return {"verification_passed": verification.passed, "verification_reason": verification.reason}


def route(state: State) -> str:
    return "pass" if state["verification_passed"] else "fail"


def report_pass(state: State) -> dict:
    print(f"PASS — verifier agrees with OpenCode's claim ({state['verification_reason']})")
    return {}


def report_fail(state: State) -> dict:
    print(f"FAIL — OpenCode claimed_done={state['claimed_done']} ({state['opencode_summary']!r}), "
          f"but the verifier disagrees: {state['verification_reason']}")
    return {}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("make_workspace", make_workspace)
    graph.add_node("run_opencode", run_opencode)
    graph.add_node("verify_result", verify_result)
    graph.add_node("report_pass", report_pass)
    graph.add_node("report_fail", report_fail)
    graph.add_edge(START, "make_workspace")
    graph.add_edge("make_workspace", "run_opencode")
    graph.add_edge("run_opencode", "verify_result")
    graph.add_conditional_edges("verify_result", route, {"pass": "report_pass", "fail": "report_fail"})
    graph.add_edge("report_pass", END)
    graph.add_edge("report_fail", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke(
        {
            "workspace": "",
            "claimed_done": False,
            "opencode_summary": "",
            "verification_passed": False,
            "verification_reason": "",
        }
    )
    assert result["claimed_done"] is True
    assert result["verification_passed"] is False


if __name__ == "__main__":
    main()
