"""06 — Verifier feedback loop.

        +-----------------------------+
        v                             |
   OpenCode task --> verifier --FAIL--+ (feed reason back, attempt += 1)
        |
      PASS -> done                     attempt == max_attempts -> give up

Same fixture and failure mode as 05, but this time a FAIL doesn't end the
graph — the verifier's failure text is folded into the *next* AgentTask's
instruction, sent back into a new OpenCode call, and the loop repeats up to
a hard `max_attempts` ceiling owned by LangGraph (not by OpenCode, and not
the same thing as a single call's `max_turns` — see
docs/failure_recovery.md).

Run: python examples/06_verifier_feedback_loop/main.py
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
MAX_ATTEMPTS = 3


class State(TypedDict):
    workspace: str
    attempt: int
    instruction: str
    verification_passed: bool
    verification_reason: str


def make_workspace(state: State) -> dict:
    workspace = tempfile.mkdtemp(prefix="feedback-loop-")
    shutil.copytree(FIXTURE, workspace, dirs_exist_ok=True)
    return {"workspace": workspace, "instruction": "The tests in this project are failing. Fix the bug."}


def run_opencode(state: State) -> dict:
    attempt = state["attempt"] + 1
    task = AgentTask(
        task_id=f"06-fix-converter-attempt-{attempt}",
        instruction=state["instruction"],
        workspace=Path(state["workspace"]),
        context={"simulate_attempt": simulate_attempt},
        max_turns=1,
    )
    OpenCodeAdapter().run(task)  # claimed_done deliberately unused: see docs/verification.md
    return {"attempt": attempt}


def verify_result(state: State) -> dict:
    verification = run_pytest(cwd=Path(state["workspace"]))
    print(f"attempt {state['attempt']}: verifier {'PASS' if verification.passed else 'FAIL'} ({verification.reason})")
    return {"verification_passed": verification.passed, "verification_reason": verification.reason}


def prepare_retry(state: State) -> dict:
    """Deterministic: folds the verifier's own words into the next instruction."""
    feedback = (
        f"The previous attempt still failed verification: {state['verification_reason']}. "
        "Look at the actual failure output, not just the fix you already tried, and correct it."
    )
    return {"instruction": feedback}


def route(state: State) -> str:
    if state["verification_passed"]:
        return "done"
    if state["attempt"] >= MAX_ATTEMPTS:
        return "give_up"
    return "retry"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("make_workspace", make_workspace)
    graph.add_node("run_opencode", run_opencode)
    graph.add_node("verify_result", verify_result)
    graph.add_node("prepare_retry", prepare_retry)
    graph.add_edge(START, "make_workspace")
    graph.add_edge("make_workspace", "run_opencode")
    graph.add_edge("run_opencode", "verify_result")
    graph.add_conditional_edges(
        "verify_result", route, {"done": END, "give_up": END, "retry": "prepare_retry"}
    )
    graph.add_edge("prepare_retry", "run_opencode")
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke(
        {
            "workspace": "",
            "attempt": 0,
            "instruction": "",
            "verification_passed": False,
            "verification_reason": "",
        }
    )
    print(f"final: {'PASS' if result['verification_passed'] else 'FAILED'} after {result['attempt']} attempt(s)")
    assert result["verification_passed"] is True
    assert result["attempt"] == 2


if __name__ == "__main__":
    main()
