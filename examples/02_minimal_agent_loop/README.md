# 02 — Minimal Agent Loop

```text
agent_step -> tool_step -> agent_step -> ... -> END
       (LLM)     (tool)    (observation)
```

The smallest illustration of what an "agent harness" actually is: a
decision-maker that can call a tool, an observation that comes back from
that tool, and a bounded loop (`MAX_TURNS`) tying the two together. Here
the "LLM" is a tiny deterministic stand-in (see example `01` for how to
wire in a real model — deliberately not repeated here, so this file stays
about the loop shape) and the only tool is a whitelisted calculator.

**Read the module docstring before anything else.** This file exists to
make the concept legible, not to be extended. Every capability a real
coding-agent loop needs beyond this — parallel tool calls, malformed
tool-call repair, context compression, sandboxed execution, streaming — is
already implemented, tested, and maintained in OpenCode (and in projects
like it; see `research/project_comparison.md`). The natural next instinct
after reading this file is usually "let's make this more capable." Do that
in `examples/03` instead, by handing the task to OpenCode's own loop, not
by growing this one.

## Run

```bash
python examples/02_minimal_agent_loop/main.py
```

## Read next

`docs/architecture.md`'s node-type table, then
`examples/03_opencode_as_node`.
