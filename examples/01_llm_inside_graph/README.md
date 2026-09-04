# 01 — LLM Inside Deterministic Control

```text
build_prompt (deterministic) -> call_llm -> shout (deterministic)
```

One LLM call, sandwiched between two ordinary functions. The LLM node has
no memory of past turns, no tools, no ability to decide what runs next —
`call_llm`'s only job is to turn a prompt into text and hand it back to the
graph. The graph, not the model, decides the next step (`shout`).

This is worth sitting with before moving to `02`: most "AI features" are
actually this pattern, not an agent. Reach for a loop (`02`) or an
autonomous kernel (`03`+) only when a single call genuinely isn't enough —
e.g. the task requires looking at the result of an action before deciding
the next one.

Runs without any API key (falls back to a clearly-labeled deterministic
stub). Set `ANTHROPIC_API_KEY` to see a real model response instead.

## Run

```bash
python examples/01_llm_inside_graph/main.py
```

## Read next

`examples/02_minimal_agent_loop`.
