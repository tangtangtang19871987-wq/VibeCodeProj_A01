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

## Session isolation: the other direction

Everything above is about OpenCode's output not leaking into LangGraph.
The reverse matters just as much: an OpenCode node must not be influenced
by history it has no business seeing — its own prior sessions, or
LangGraph's own accumulated thread/checkpoint history. Three separate
mechanisms enforce this, and all three have to hold; any one alone is not
enough.

1. **A fresh OpenCode session every call, no exceptions.** Both real
   backends in `src/opencode_adapter.py` get this independently:
   `CLIBackend` never passes `--continue`/`-c`, `--session`/`-s`, or
   `--fork` (`research/sources.md`), with an `assert` in
   `CLIBackend.execute()` as a tripwire against ever adding one back by
   accident; `SDKBackend` calls `client.session.create()` itself at the
   start of every `execute()` and never accepts an externally supplied
   session id — its docstring says so explicitly, and it's the more
   visible of the two guarantees, since you get back a real `Session`
   object with a fresh id each time rather than relying on the *absence*
   of a flag. Either way, `AgentTask` (`src/contracts.py`) has no
   `session_id` field at all: there's no argument slot for a caller to
   accidentally hand OpenCode a stale session to continue. If a real task
   genuinely needs multi-turn memory of *its own* earlier attempts
   (example `06`'s retry loop), that memory is threaded explicitly through
   `AgentTask.instruction` (the verifier's own failure text, folded in by
   a deterministic node) — never through OpenCode's session store.
2. **A workspace is scoped to one logical task, never shared across
   unrelated ones.** OpenCode can read anything sitting in its workspace
   directory, so stale files from a *different* task in the same directory
   are a second history-leak vector even with a fresh session — it might
   read an old note, a half-applied fix, or a previous task's log and let
   that quietly bias its investigation. The rule: a workspace may be
   reused across repeated attempts at the *same* task (example `06`
   reuses one workspace across its retry loop on purpose — that's the
   task's own evolving state, not contamination), but a genuinely
   different task always gets a fresh one (`tempfile.mkdtemp()`, as in
   examples `04`, `08`, `09`). Never point two unrelated `AgentTask`s at
   the same workspace.
3. **`AgentTask.context` is built fresh by the calling node, never copied
   from LangGraph's own accumulated state.** A node with access to a
   checkpointed thread's full history should still hand OpenCode only the
   two or three fields relevant to *this* task — see "Practical guidance"
   below. Otherwise the LangGraph side becomes the history-leak vector
   instead of OpenCode's session store.

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
