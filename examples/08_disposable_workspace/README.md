# 08 — Disposable Workspace

```text
create workspace -> hand to OpenCode -> verify -> promote or discard
```

Runs the same graph twice against two different (simulated) OpenCode
outcomes. The graph — not OpenCode — creates each temp workspace, and the
graph — again, not OpenCode — decides what happens to it afterward, based
on `src/verifier.py`'s result:

- A task that actually gets fixed has its workspace copied into
  `promoted/` and the temp directory removed.
- A task OpenCode can't complete has its temp directory removed with
  nothing copied out — no half-finished or failed attempt leaks into
  anything permanent.

This is the practical alternative to building a custom permission system
around every tool call OpenCode might make (`docs/sandboxing.md`): give it
a disposable place to work freely, and let verification — not trust in the
agent's judgment — decide what survives.

## Run

```bash
python examples/08_disposable_workspace/main.py
```

`promoted/` is created next to this README as example output; it's
git-ignored and safe to delete between runs.

## Read next

`examples/09_domain_workflow_with_agent_escape_hatch` — the capstone.
