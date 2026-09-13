# ADR 0013: Harness injection/use hook selection deferred to Milestone 5

Status: Accepted

## Context

Section 25 item 10 asks which harness can provide reliable injection/use
hooks. This only matters once `memory_feedback`'s `injected`/`used` event
types have a real caller (Milestone 5); answering it now would be a guess
disconnected from implementation.

## Decision

Defer this decision to Milestone 5. Until then, `injected`/`used` evidence
comes only from whatever explicit `memory_feedback` calls a harness or
human chooses to make — no adapter-specific hook exists, and none is
assumed. Milestone 5 will evaluate whichever of OpenCode, Copilot CLI, or
Claude Code exposes a stable, documented hook for "this tool result/context
was included in the model request" at that time, and record the outcome in
a follow-up ADR.

## Consequences

No Milestone 0–4 code depends on a specific harness's internal hook
surface, keeping the daemon harness-agnostic per principle 4.4.

## Alternatives considered

Guessing now and building against one harness's current (unstable) internals:
rejected as premature coupling to an implementation detail outside this
project's control.
