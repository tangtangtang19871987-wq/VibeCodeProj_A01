# 07 — Context Isolation

OpenCode's stand-in here deliberately fails its own check twice before
passing, generating a chunk of verbose "build output" on each attempt —
standing in for the shell/tool chatter a real debugging session produces.
The example builds the resulting graph state two different ways from the
same `AgentResult` and prints the size of each:

- **isolated** (what every other example in this repo does): `{summary,
  log_path}` — a couple hundred bytes, regardless of how much OpenCode did
  internally.
- **leaked** (an anti-pattern, computed here only to measure it — never
  wired into a real graph): the full transcript inlined into state.

On a typical run the leaked version is 50x+ larger, and that ratio only
gets worse the longer and more turn-heavy a real OpenCode session runs.
See `docs/context_management.md` for the mechanism this borrows from (Deep
Agents' isolated-subagent state, OpenHands' static/dynamic prompt split).

The full transcript is never gone in the isolated version — `log_path`
still points at it. The point is that nothing downstream pays for it
unless a node explicitly opts in by reading the file.

## Run

```bash
python examples/07_context_isolation/main.py
```

## Read next

`examples/08_disposable_workspace`.
