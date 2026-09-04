# 03 — OpenCode as a LangGraph Node

```text
LangGraph
   |
OpenCode
   |
LangGraph
```

The graph here is intentionally boring: `prepare_task` (deterministic) ->
`run_opencode` (the entire boundary crossing) -> `report` (deterministic).
`run_opencode` is the only node in this whole repository that's allowed to
know an `AgentTask`/`AgentResult` exist — everything else just sees
ordinary graph state.

Notice what's *not* here: no retry, no verification, no inspection of what
OpenCode did internally. Those are separate concerns, covered in `05` and
`06`. This example exists purely so the shape of "OpenCode as a node" is
clear before those concerns get added on top of it.

By default this runs against `src.opencode_adapter`'s `FakeBackend` (a
small, honest test double — see the docstring in
`src/opencode_adapter.py`), so it works without installing OpenCode or a
model API key. To use the real CLI instead:

```bash
OPENCODE_ADAPTER_BACKEND=cli python examples/03_opencode_as_node/main.py
```

## Run

```bash
python examples/03_opencode_as_node/main.py
```

## Read next

`examples/04_autonomous_coding_worker`.
