"""07 — Context isolation.

Demonstrates docs/context_management.md's central claim with numbers, not
just an assertion: OpenCode's internal transcript can be large, and the
LangGraph state that results from a task does not have to grow with it.

The fixture task is set up so OpenCode "fails" its own check twice before
succeeding, each attempt appending a chunk of realistic-looking shell
output (a stand-in for what a real debugging session's stdout looks like)
to its session log. We then build the graph state two different ways from
the *same* AgentResult:

  - `isolated_result` (what every other example in this repo does): only
    `summary` and `log_path` cross into state.
  - `leaky_result` (an anti-pattern, computed here ONLY for comparison —
    never wired into an actual graph): the full transcript inlined into
    state, as if a node had done `state["opencode_transcript"] = log.read()`.

Run: python examples/07_context_isolation/main.py
"""

from __future__ import annotations

import glob
import json
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

_VERBOSE_CHUNK = (
    "resolving dependencies...\n" + "  checked candidate package\n" * 40
    + "running build steps...\n" + "  compiling module\n" * 40
)


def _simulate_verbose_attempt(workspace: Path, attempt: int, last_error: str) -> str:
    # Stands in for OpenCode's own shell/tool output during one attempt —
    # this is exactly the kind of thing that's fine to have a LOT of on
    # disk and not fine to have a lot of in graph state.
    (workspace / f"attempt_{attempt}.log").write_text(_VERBOSE_CHUNK)
    return f"ran a verbose build/debug cycle ({len(_VERBOSE_CHUNK)} chars of output)"


class State(TypedDict):
    workspace: str
    isolated_state_repr: str
    leaky_state_repr: str


def run_and_isolate(state: State) -> dict:
    workspace = Path(state["workspace"])
    task = AgentTask(
        task_id="07-context-isolation",
        instruction="Investigate and resolve the (simulated) issue.",
        workspace=workspace,
        context={
            "simulate_attempt": _simulate_verbose_attempt,
            # fails twice, then passes: three verbose attempts logged
            "check_command": [
                sys.executable,
                "-c",
                "import sys, glob; sys.exit(0 if len(glob.glob('attempt_*.log')) >= 3 else 1)",
            ],
        },
        max_turns=5,
    )
    result = OpenCodeAdapter().run(task)

    # The pattern this repo actually uses: only a summary and a pointer.
    isolated_state = {"summary": result.summary, "log_path": str(result.log_path)}

    # The anti-pattern, computed here ONLY to measure the difference —
    # never do this in a real node. `attempt_*.log` stands in for
    # everything OpenCode read/ran/observed internally across its retries.
    transcript_files = sorted(glob.glob(str(workspace / "attempt_*.log")))
    full_transcript = "".join(Path(f).read_text() for f in transcript_files)
    leaky_state = {"summary": result.summary, "full_transcript": full_transcript}

    return {
        "isolated_state_repr": json.dumps(isolated_state),
        "leaky_state_repr": json.dumps(leaky_state),
    }


def report(state: State) -> dict:
    isolated_size = len(state["isolated_state_repr"])
    leaky_size = len(state["leaky_state_repr"])
    print(f"graph state size if isolated (summary + pointer only): {isolated_size} bytes")
    print(f"graph state size if leaked (full transcript inlined):  {leaky_size} bytes")
    print(f"ratio: {leaky_size / isolated_size:.1f}x larger")
    return {}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("run_and_isolate", run_and_isolate)
    graph.add_node("report", report)
    graph.add_edge(START, "run_and_isolate")
    graph.add_edge("run_and_isolate", "report")
    graph.add_edge("report", END)
    return graph.compile()


def main() -> None:
    workspace = tempfile.mkdtemp(prefix="context-isolation-")
    try:
        app = build_graph()
        result = app.invoke({"workspace": workspace, "isolated_state_repr": "", "leaky_state_repr": ""})
        assert len(result["leaky_state_repr"]) > len(result["isolated_state_repr"])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
