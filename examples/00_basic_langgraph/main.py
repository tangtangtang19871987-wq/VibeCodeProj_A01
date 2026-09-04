"""00 — Deterministic LangGraph.

START -> normalize -> count_words -> END

No LLM, no agent, no OpenCode. This is the floor everything else in this
repository is built on: a graph is just typed state flowing through plain
functions, with the graph itself deciding what runs next. Every later
example replaces one of these plain-function nodes with something more
interesting (an LLM call, then a toy agent loop, then OpenCode) without
changing this basic shape.

Run: python examples/00_basic_langgraph/main.py
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    raw_text: str
    normalized_text: str
    word_count: int


def normalize(state: State) -> dict:
    return {"normalized_text": state["raw_text"].strip().lower()}


def count_words(state: State) -> dict:
    words = state["normalized_text"].split()
    return {"word_count": len(words)}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("normalize", normalize)
    graph.add_node("count_words", count_words)
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "count_words")
    graph.add_edge("count_words", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke({"raw_text": "  LangGraph Is The Control Plane  "})
    print(result)
    assert result["word_count"] == 5


if __name__ == "__main__":
    main()
