# Context Management

## The problem

An OpenCode session accumulates a lot of internal state fast: every file it
reads, every shell command and its full stdout/stderr, every intermediate
plan, every failed attempt. None of that is inherently useful to the
LangGraph state — and if it all flows back into graph state, three things
go wrong at once:

1. **Prompt bloat.** Any LLM node downstream of the OpenCode node now pays
   for (and gets distracted by) kilobytes of shell output it never needed.
2. **Checkpoint bloat.** LangGraph checkpointers persist state; a workspace's
   full debugging transcript is not something you want serialized into
   every checkpoint of an otherwise small workflow.
3. **Coupling.** Nodes downstream start depending on the *shape* of
   OpenCode's internal chatter, which is not a stable interface.

## The rule: context quarantine

> Only compact task/result information crosses the LangGraph <-> OpenCode
> boundary. Everything else stays on disk, addressed by a pointer.

Concretely, in `src/contracts.py`:

- `AgentTask.context` is a small dict the caller builds on purpose — not
  "whatever's in state." A node calling into OpenCode should construct this
  explicitly, the same way you'd construct a request body for an API call.
- `AgentResult.summary` is a short string. `AgentResult.log_path` is a
  `Path` to the *full* transcript OpenCode produced, written to the
  workspace, never inlined into `AgentResult` itself.

`examples/07_context_isolation` makes this concrete and measurable: it runs
the same task twice, once returning the full transcript into graph state and
once returning only `(summary, log_path)`, and prints the size of the
resulting state object each time. The point isn't that reading the full log
is impossible — `log_path` is right there — it's that the graph never pays
for it unless a node explicitly opts in by reading the file.

## Precedent this borrows from

This is not a new idea — it is the same shape as two mechanisms already
proven in the wild (see `research/project_comparison.md` for full detail):

- **Deep Agents' isolated subagents**
  (`libs/deepagents/deepagents/middleware/subagents.py`): a `mode="isolated"`
  subagent is handed a *fresh* state —
  `messages=[HumanMessage(description)]`, none of the parent's history — and
  only a single `ToolMessage` is merged back into the parent. The parent
  never sees the subagent's internal turns. `mode="fork"` is the deliberate
  opposite: full inheritance, for when a subagent genuinely needs the
  parent's context. Two named policies, chosen per call, not one default
  that leaks everything.
- **The OpenHands SDK's static/dynamic prompt split**
  (`openhands/sdk/agent/base.py`): `static_system_message` (cacheable, built
  once) is kept separate from `dynamic_context` (skills, secrets, runtime
  info, assembled per conversation). It's driven by prompt-caching economics
  there, but the underlying discipline — never let situational detail bleed
  into what's supposed to be stable — is the same discipline this repo
  applies at the LangGraph/OpenCode boundary.

## What this is *not*

This is not a context-compression system. Summarization/condensing (Deep
Agents' `SummarizationMiddleware`, OpenHands' `condenser`) is a different,
already-solved problem: how to keep *one* agent's own history usable as it
grows. Context quarantine is about the boundary *between* systems — deciding
what's allowed to cross it at all, not how to shrink what's already inside.
Don't build a condenser here; if a task genuinely needs one, that's OpenCode's
job internally.

## Practical guidance

- Treat every `AgentTask.context` you construct as if it were going over a
  network call to a different service — because architecturally, it is.
- Never pass `state` (the whole LangGraph state dict) into an OpenCode task
  wholesale. Pick the two or three fields that matter.
- If a downstream node needs something from OpenCode's internal process
  (not just its final result), that's a sign the task boundary is drawn in
  the wrong place — split it into two bounded tasks instead of leaking
  context to work around it.
