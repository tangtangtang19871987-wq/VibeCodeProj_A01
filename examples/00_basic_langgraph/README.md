# 00 — Basic LangGraph

```text
START -> normalize -> count_words -> END
```

The simplest possible LangGraph program: two plain Python functions, wired
together, running against typed state. Nothing here is an "agent" — there
is no LLM, no tool call, no loop. That's the point: before anything in this
repository gets autonomous, it's worth seeing exactly what the deterministic
substrate looks like with none of that yet.

Every node in every later example — an LLM call (`01`), a toy agent loop
(`02`), an OpenCode-backed worker (`03`+) — is still, from the graph's point
of view, just a function that takes state in and returns an update. That
uniformity is why LangGraph can be the control plane for all of them without
needing to special-case "the autonomous one."

## Run

```bash
python examples/00_basic_langgraph/main.py
```

## Read next

`docs/architecture.md`, then `examples/01_llm_inside_graph`.
