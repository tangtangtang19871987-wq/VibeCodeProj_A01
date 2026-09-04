"""01 — LLM inside deterministic control.

deterministic -> LLM -> deterministic

The LLM node is a single call with no loop, no tool use, no autonomy: it
takes a prompt, returns text, and the graph decides what happens with that
text. This is the "ordinary LLM node" column from docs/architecture.md's
node-type table — contrast it with example 02 (a loop) and example 03 (a
whole autonomous kernel as a single node).

If ANTHROPIC_API_KEY is set and the `anthropic` package is installed, this
calls a real model. Otherwise it falls back to a small deterministic stub
so the example still demonstrates the *graph wiring* without requiring
credentials — the fallback is clearly labeled in its own output, never
silently pretended to be a real model response.

Run: python examples/01_llm_inside_graph/main.py
"""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    topic: str
    prompt: str
    headline: str
    shout_headline: str


def build_prompt(state: State) -> dict:
    prompt = f"Write a single short, punchy headline about: {state['topic']}. Reply with only the headline."
    return {"prompt": prompt}


def call_llm(state: State) -> dict:
    return {"headline": _call_llm(state["prompt"])}


def shout(state: State) -> dict:
    return {"shout_headline": state["headline"].upper() + "!"}


def _call_llm(prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic  # noqa: PLC0415

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as exc:  # pragma: no cover - network/credential path
            return f"[llm call failed, using stub: {exc}] Stub headline for: {prompt}"
    # No credentials available in this environment. This stub exists only
    # so the example runs out of the box; it is not a model and should not
    # be mistaken for one.
    return f"[STUB - no ANTHROPIC_API_KEY set] Deterministic Stand-In Headline About {prompt.split(':')[-1].strip()}"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("call_llm", call_llm)
    graph.add_node("shout", shout)
    graph.add_edge(START, "build_prompt")
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "shout")
    graph.add_edge("shout", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke({"topic": "bounded autonomy in agent systems"})
    print(result["headline"])
    print(result["shout_headline"])


if __name__ == "__main__":
    main()
