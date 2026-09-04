"""09 — Domain workflow with an autonomous escape hatch. The capstone.

deterministic domain workflow
        |
   unusual problem?
      /      \\
    no        yes
    |          |
 continue   OpenCode
              |
           verifier
              |
           continue

A small order-intake pipeline: parse an order file, validate it against a
known schema, compute its total. Most orders are well-formed and never go
near OpenCode — `classify` routes them straight through deterministic
code. One order in this example is malformed in a shape the validator was
never written to special-case (wrong key names, values with currency
symbols as strings) — not a rule the deterministic code should grow an
`elif` for, but also not something to give up on. That one order is
escalated to a bounded OpenCode task, whose output is re-validated by the
*same* deterministic validator used for classification (never trusted
outright — docs/verification.md) before the pipeline continues.

The autonomous agent is not the workflow. It's a capability the workflow
reaches for in one narrow, well-defined circumstance. Everything else in
this example — parsing, schema validation, total computation, routing — is
exactly as deterministic as example 00.

Run: python examples/09_domain_workflow_with_agent_escape_hatch/main.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, TypedDict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from langgraph.graph import END, START, StateGraph

from src.contracts import AgentTask, VerificationResult
from src.opencode_adapter import OpenCodeAdapter

ORDERS_FILE = "order.json"


# ---- deterministic domain logic -------------------------------------------------


def validate_order(order_path: Path) -> VerificationResult:
    """The one function that decides whether an order is well-formed.

    Used twice: once by `classify` to decide whether a case is "unusual"
    at all, and again after the escape hatch to independently check
    OpenCode's repair. Same rules, same code, in both places — that's what
    makes the post-repair check a real verification and not a rubber stamp.
    """
    try:
        data = json.loads(order_path.read_text())
    except json.JSONDecodeError as exc:
        return VerificationResult(passed=False, reason=f"invalid JSON: {exc}")

    if "items" not in data or not isinstance(data["items"], list) or not data["items"]:
        return VerificationResult(passed=False, reason="missing or empty 'items' list")
    for item in data["items"]:
        if not isinstance(item.get("sku"), str):
            return VerificationResult(passed=False, reason=f"item missing string 'sku': {item}")
        if not isinstance(item.get("qty"), int) or item["qty"] <= 0:
            return VerificationResult(passed=False, reason=f"item has invalid integer 'qty': {item}")
        if not isinstance(item.get("unit_price"), (int, float)) or item["unit_price"] < 0:
            return VerificationResult(passed=False, reason=f"item has invalid numeric 'unit_price': {item}")
    if not isinstance(data.get("currency"), str):
        return VerificationResult(passed=False, reason="missing string 'currency'")
    return VerificationResult(passed=True, reason="schema OK", details={"order": data})


def compute_total(order_path: Path) -> tuple[float, str]:
    data = json.loads(order_path.read_text())
    total = sum(item["qty"] * item["unit_price"] for item in data["items"])
    return total, data["currency"]


# ---- the fake backend's stand-in for "OpenCode reshapes an odd order" ----------


def _repair_order_shape(workspace: Path, attempt: int, last_error: str) -> str:
    order_path = workspace / ORDERS_FILE
    raw = json.loads(order_path.read_text())
    items = raw.pop("products", raw.get("items", []))
    fixed_items = []
    for item in items:
        fixed_items.append(
            {
                "sku": item["sku"],
                "qty": int(item["qty"]),
                "unit_price": float(str(item["unit_price"]).replace("$", "").replace(",", "")),
            }
        )
    raw["items"] = fixed_items
    order_path.write_text(json.dumps(raw))
    return "renamed 'products' -> 'items' and coerced qty/unit_price to numbers"


# ---- the graph -------------------------------------------------------------------


class State(TypedDict):
    workspace: str
    route: str
    total: float
    currency: str
    note: str


def classify(state: State) -> dict:
    verification = validate_order(Path(state["workspace"]) / ORDERS_FILE)
    return {"route": "normal" if verification.passed else "unusual", "note": verification.reason}


def route_after_classify(state: State) -> str:
    return state["route"]


def escape_hatch_opencode(state: State) -> dict:
    """The ONLY node in this workflow that touches OpenCode."""
    workspace = Path(state["workspace"])
    task = AgentTask(
        task_id="09-repair-order-shape",
        instruction=(
            f"order.json doesn't match the expected schema ({state['note']}). "
            "Reshape it to match: {items: [{sku: str, qty: int, unit_price: number}], currency: str}."
        ),
        workspace=workspace,
        context={"simulate_attempt": _repair_order_shape},
        max_turns=1,
        permissions="read_write",
    )
    OpenCodeAdapter().run(task)  # claimed_done unused: verify_repair is what decides
    return {}


def verify_repair(state: State) -> dict:
    verification = validate_order(Path(state["workspace"]) / ORDERS_FILE)
    return {"route": "normal" if verification.passed else "manual_review", "note": verification.reason}


def route_after_repair(state: State) -> str:
    return state["route"]


def finish_order(state: State) -> dict:
    total, currency = compute_total(Path(state["workspace"]) / ORDERS_FILE)
    return {"total": total, "currency": currency, "note": "completed"}


def manual_review(state: State) -> dict:
    return {"note": f"escalating to a human: repair did not produce a valid order ({state['note']})"}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("classify", classify)
    graph.add_node("escape_hatch_opencode", escape_hatch_opencode)
    graph.add_node("verify_repair", verify_repair)
    graph.add_node("finish_order", finish_order)
    graph.add_node("manual_review", manual_review)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify", route_after_classify, {"normal": "finish_order", "unusual": "escape_hatch_opencode"}
    )
    graph.add_edge("escape_hatch_opencode", "verify_repair")
    graph.add_conditional_edges(
        "verify_repair", route_after_repair, {"normal": "finish_order", "manual_review": "manual_review"}
    )
    graph.add_edge("finish_order", END)
    graph.add_edge("manual_review", END)
    return graph.compile()


def run_order(app, order: dict[str, Any]) -> dict:
    workspace = tempfile.mkdtemp(prefix="order-")
    (Path(workspace) / ORDERS_FILE).write_text(json.dumps(order))
    try:
        return app.invoke({"workspace": workspace, "route": "", "total": 0.0, "currency": "", "note": ""})
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> None:
    app = build_graph()

    well_formed_order = {
        "items": [{"sku": "WIDGET", "qty": 3, "unit_price": 12.5}],
        "currency": "USD",
    }
    unusual_order = {
        "products": [{"sku": "WIDGET", "qty": "3", "unit_price": "$12.50"}],
        "currency": "USD",
    }

    for label, order in [("well-formed", well_formed_order), ("unusual-shape", unusual_order)]:
        result = run_order(app, order)
        if result["note"] == "completed":
            print(f"[{label}] completed: total={result['total']} {result['currency']}")
        else:
            print(f"[{label}] {result['note']}")

    normal_result = run_order(app, well_formed_order)
    unusual_result = run_order(app, unusual_order)
    assert normal_result["total"] == 37.5
    assert unusual_result["total"] == 37.5  # same order, reshaped via the escape hatch, same answer


if __name__ == "__main__":
    main()
