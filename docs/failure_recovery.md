# Failure Recovery

## Two different kinds of failure

1. **OpenCode's own internal failures** — a shell command errors, a patch
   doesn't apply, a test fails on the first try. This is exactly what
   OpenCode's internal loop already exists to handle: investigate, adjust,
   retry, inside its own turn budget. This repo does not intervene here and
   should not try to — see `research/project_comparison.md` for why
   reimplementing traceback-driven repair would be pure duplication.
2. **Task-level failure** — OpenCode reports `claimed_done=True` (or exhausts
   its turns without doing so) and the external verifier says FAIL anyway.
   This is where LangGraph takes over.

Conflating the two is the mistake to avoid: don't build a LangGraph-level
retry loop that re-runs OpenCode from scratch on every internal hiccup (that
duplicates what OpenCode already does, badly, from the outside). Only
retry at the task level, driven by a verifier's result.

## The task-level retry loop

```text
        +-----------------------------+
        v                             |
   OpenCode task --> verifier --FAIL--+ (feed reason back, attempt += 1)
        |                             
      PASS -> continue                 attempt == max_attempts -> give up, escalate
```

`examples/06_verifier_feedback_loop` implements this: a LangGraph-owned
counter (not something OpenCode tracks about itself) increments each time
verification fails, `VerificationResult.reason` is folded into the next
`AgentTask.instruction` sent to the *same* OpenCode session (so it isn't
starting from zero context each time), and the loop has a hard
`max_attempts` ceiling enforced by the conditional edge, independent of
whatever `AgentTask.max_turns` bounds a single OpenCode call to internally.

## Giving up is a real outcome, not a bug

When `max_attempts` is reached and verification still fails, the graph
should route to an explicit "needs human" or "task failed" terminal state —
not loop forever, and not silently report success. This mirrors Open SWE's
Reviewer gate: a PR that still doesn't pass loops back to planning rather
than being force-merged, and there's an implicit ceiling on how long that
loop runs before a human has to look at it (`research/project_comparison.md`).

## Where checkpoints matter

Because each task-level attempt is its own `AgentTask`/`AgentResult`/
`VerificationResult` triple, and the retry count lives in ordinary
LangGraph state, a checkpointer (`langgraph.checkpoint`) makes the whole
retry loop resumable across process restarts for free — if the process
running the graph crashes mid-loop, resuming from the last checkpoint
picks up the attempt counter and the last verifier reason exactly where
they were, without re-running attempts that already happened. None of the
examples in this repo wire up a persistent checkpointer (kept out to stay
teaching-sized), but the state shape is checkpointer-ready by construction:
everything that matters is a small, serializable value in `State`, never a
live subprocess handle or an open file descriptor.

## Distinguishing "OpenCode failed" from "the task was impossible"

`AgentResult.error` (`src/contracts.py`) is set when the OpenCode call
itself errored (timeout, crash, non-zero exit with no usable output) as
opposed to running to completion and simply not satisfying the verifier.
Route these differently: an `error` is worth surfacing immediately (it may
indicate a bad `AgentTask`, not a hard problem), whereas a verifier FAIL
after a clean run is exactly the case the feedback loop is for.
