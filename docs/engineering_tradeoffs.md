# Engineering Tradeoffs

Honest answers to "where does this architecture work well, and where
doesn't it."

## Where it works well

- **Tasks with a fuzzy middle and a checkable end.** "Fix this failing
  test," "make this script handle a new input format," "resolve this
  dependency conflict" — hard to formalize as deterministic steps, easy to
  verify objectively once done. This is the sweet spot: let OpenCode figure
  out the fuzzy part, let a deterministic verifier confirm the end.
- **Systems that are mostly deterministic already.** If 95% of your
  workflow is well-understood domain logic and 5% is "sometimes we hit a
  case nobody anticipated," `examples/09`'s escape-hatch pattern adds
  autonomy exactly where it's needed without rewriting the other 95%.
- **Anything where you'd otherwise hand-write a coding-agent loop.** If the
  alternative is building your own shell-execution-plus-retry harness, using
  OpenCode is very likely cheaper and better-tested than the bespoke
  version, per `research/project_comparison.md`.

## Where it works poorly

- **Purely deterministic tasks.** If you can write the steps down, write
  them down. Routing a deterministic task through an autonomous kernel adds
  latency, cost, and a new failure mode (the agent "helpfully" doing
  something you didn't ask for) for zero benefit.
- **Latency-sensitive paths.** Every OpenCode node is at minimum one
  subprocess invocation plus one or more model round-trips inside it. A
  synchronous user-facing request path is usually the wrong place for this;
  batch/background workflows tolerate it far better.
- **Tasks where "success" can't be checked deterministically.** If you
  can't write a verifier for it, this architecture's central safety
  property — independent verification — doesn't apply, and you're back to
  trusting the agent's own claim. That's a real limitation, not something
  to paper over with a "vibes-based" verifier.
- **Very high-frequency small tasks.** The fixed overhead of spinning up an
  autonomous kernel per call adds up. Batch related work into fewer, larger
  bounded tasks instead of many tiny ones.
- **Tasks needing cross-task memory the isolated-workspace model doesn't
  give you for free.** Disposable workspaces (`sandboxing.md`) are great for
  independence but mean nothing persists between tasks unless you
  explicitly design a promotion path — see `examples/08`.

## Cost and latency accounting

Three things you're paying for, stacked: (1) the LLM calls inside
OpenCode's own loop, scaled by however many internal turns it takes and
bounded by `AgentTask.max_turns`; (2) the verifier — usually cheap
(running a test suite) but not free; (3) retry attempts on the outer
feedback loop (`failure_recovery.md`), each of which repeats (1) and (2).
Set `max_turns` and `max_attempts` deliberately; they are the two knobs
that most directly bound total cost per task.

## The "agent first, then harden" path

The brief's guidance: if a task is hard to formalize, let OpenCode solve it
first. If repeated runs reveal a stable pattern — the same handful of
fixes, the same shape of input every time — extract *that specific pattern*
into a deterministic LangGraph node later, and stop routing it to OpenCode
at all. `examples/09`'s escape hatch is the concrete mechanism for this:
watch what falls through the deterministic router into the OpenCode branch
over time, and when a category of "unusual" case turns out not to be
unusual, move its handling into the deterministic side of the router. This
is a migration path, not a one-time design decision — expect the boundary
between the two sides of `09`'s router to move over the life of a real
system.

## Where this repository intentionally stops

This is a teaching repository, not a production framework. It does not
include: a persistent checkpointer wired up end-to-end, container-based
sandbox enforcement, observability/tracing integration, or a real
multi-provider LLM abstraction. Each of those is a reasonable next step for
a production system built on this pattern, and each is called out at the
point in `docs/` where it would matter, rather than built here and left
half-finished.
