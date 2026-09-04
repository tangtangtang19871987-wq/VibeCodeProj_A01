# Project Comparison

Structured notes on five projects, each answered against the same ten
questions from the project brief. Sources are in `sources.md`; this file is
the synthesis.

Legend: **D** = deterministic, **A** = autonomous.

---

## LangGraph

1. **Deterministic:** The entire graph topology — nodes, edges, conditional
   routing functions, `Command(goto=...)` transitions — is defined in code
   and executed by a fixed Pregel-style scheduler. Determinism is the
   product's whole premise.
2. **Autonomous:** Nothing, by itself. LangGraph has no opinion about what a
   node does; a node can call an LLM, run a subprocess, or just be a `def`.
   Autonomy only exists if you put it inside a node.
3. **State:** A typed `State` object (TypedDict/Pydantic/dataclass) threaded
   through every node; reducers control how concurrent branches merge
   updates into the same key.
4. **Context:** Whatever you put in `State`. LangGraph doesn't manage
   "context" as a concept — that's left entirely to the graph author, which
   is exactly why context isolation must be designed, not assumed (see
   `docs/context_management.md`).
5. **Tools:** Not a first-class concept at the graph level — tools live
   inside whichever node/agent calls them (e.g. via `create_react_agent` or
   a hand-written node).
6. **Autonomy bounding:** `recursion_limit`, `interrupt()` for
   human-in-the-loop pausing, checkpointers for resuming from a known-good
   state, and the fact that routing is a plain Python function you write —
   bounding is available but opt-in.
7. **Failures:** Checkpointers make a node's failure resumable rather than
   fatal to the whole run; retries are the graph author's responsibility
   (LangGraph provides the primitives, not a policy).
8. **Success:** Whatever the graph author's conditional edges say it is —
   LangGraph has no built-in notion of "done."
9. **Borrow:** `Command` for explicit state+routing transitions,
   `interrupt()`/checkpointer for pause-resume, conditional edges as the
   place verification result gets to change control flow.
10. **Unnecessary to reimplement:** N/A — this *is* the orchestration layer
    this repo uses, not something to route around.

## OpenCode (sst/opencode)

1. **Deterministic:** The client/server split, session storage, and
   permission-rule engine (what a tool call is allowed to do without
   prompting a human) are all deterministic scaffolding around the model.
2. **Autonomous:** The actual coding loop — read code, run shell commands,
   edit files, re-run, interpret output, retry — happens entirely inside
   OpenCode's own agent loop once a task is handed to it. That loop is a
   capability this repo deliberately does not reimplement.
3. **State:** SQLite-backed sessions on the server side; a session can be
   continued (`--continue`/`-c`) or forked (`--fork`) rather than restarted
   from scratch.
4. **Context:** Whatever accumulates inside that one OpenCode session —
   file reads, shell output, edit diffs. From LangGraph's point of view this
   is opaque and, by design, should stay that way (see "context quarantine").
5. **Tools:** A fixed built-in set (`bash`, file read/write/edit, `glob`,
   `grep`, LSP diagnostics, `fetch`) plus MCP servers for extensions —
   exposed to the model, not to the orchestrator.
6. **Autonomy bounding:** A named-agent permission tier (`build` = full
   access, `plan` = read-only with bash gated behind a prompt) and, in
   non-interactive mode, permission rules that **auto-deny** prompts like
   `question`/`plan_enter`/`plan_exit` rather than hanging waiting for a
   human that isn't there — headless runs fail closed, not open.
7. **Failures:** Left to the model's own loop to notice (failing test,
   traceback) and retry within the session; the CLI surfaces this only as
   `process.exitCode = 1` plus an event stream, i.e. OpenCode reports
   failure, it doesn't get to redefine success.
8. **Success:** OpenCode does not decide this for the outer system — it
   returns text/events and an exit code. Whether the *task* succeeded is
   for whatever invoked it to check. This is precisely the seam this
   repo's `verifier.py` sits in.
9. **Borrow:** headless invocation as a single subprocess call
   (`opencode run`), the `build`/`plan` permission-tier idea as inspiration
   for "wide observation, narrow mutation," fail-closed permission
   defaults for unattended runs.
10. **Unnecessary to reimplement:** the coding-agent loop itself — file
    search, shell execution, patch application, traceback-driven repair,
    iterative self-correction. This is the single biggest reason OpenCode
    exists inside this architecture at all.

## Deep Agents (langchain-ai/deepagents)

1. **Deterministic:** The middleware stack composition
   (`create_deep_agent`) is fixed and inspectable; each middleware's tool
   contract (e.g. `write_todos`, `ls`/`read_file`/...) is deterministic
   Python.
2. **Autonomous:** The single underlying `create_agent` ReAct-style loop —
   one big agent deciding, each step, whether to call a tool or finish —
   with a **default `recursion_limit` of 9,999**, i.e. autonomy is barely
   bounded by default; the model is trusted to stop itself.
3. **State:** LangGraph state, extended with a virtual filesystem
   (`FilesystemState.files`, a delta-channel dict with periodic snapshots)
   and a todo list, both just more state keys.
4. **Context:** The interesting mechanism is subagent isolation:
   `SubAgentMiddleware`'s `task` tool builds a **fresh** state for
   `mode="isolated"` subagents — `messages=[HumanMessage(description)]`,
   nothing else from the parent — and only a single `ToolMessage` crosses
   back. `mode="fork"` instead inherits (a filtered copy of) the parent's
   message history. Two explicit, named context-crossing policies rather
   than one default.
5. **Tools:** Composed per-middleware — filesystem tools, `task` (subagent
   delegation), `write_todos` (planning) — all ordinary LangChain tools
   attached to one agent.
6. **Autonomy bounding:** Mostly soft: a very high recursion ceiling,
   `HumanInTheLoopMiddleware` for opt-in interrupts, and
   `SandboxBackendProtocol` for scoping *where* `execute` can act — but no
   external verifier is wired in by default. Boundedness here is
   structural (isolated context) more than a hard stop.
7. **Failures:** `PatchToolCallsMiddleware` repairs malformed tool calls;
   beyond that, failure handling is the agent's own ReAct loop retrying.
8. **Success:** The agent itself, via `FinishTool`-equivalent behavior —
   there's no separate verifier stage in the framework itself.
9. **Borrow directly:** the isolated vs. forked subagent-context pattern
   (this repo's `docs/context_management.md` and example `07` are modeled
   on it almost 1:1), the `StateBackend`/`FilesystemBackend`/
   `CompositeBackend` split for choosing where a workspace actually lives,
   and `write_todos` as a cheap "make the plan legible" primitive.
10. **Unnecessary to reimplement:** the generic ReAct tool-calling loop, the
    virtual filesystem tool surface, malformed-tool-call repair — all
    solved problems that would be pure duplication to rebuild around
    OpenCode, which already has its own equivalents internally.

## OpenHands Software Agent SDK

1. **Deterministic:** `Conversation`/`EventStream` plumbing, the
   `Workspace` abstraction (local vs. containerized vs. remote — swap the
   implementation, keep the agent code identical), and `AgentBase.verify()`
   which enforces that a resumed agent's tool set may only grow, never
   shrink, relative to what was persisted.
2. **Autonomous:** `AgentBase.step(conversation, on_event, on_token)` — the
   abstract core loop, explicitly documented as mutating state in place —
   is where the model decides what `Action` to take next; concrete agents
   subclass this.
3. **State:** Event-sourced. Every action/observation is an event on the
   stream; conversation state is derivable from replaying events, which is
   what makes resumption and persistence coherent.
4. **Context:** Deliberately split into `static_system_message` (cacheable,
   built once) and `dynamic_context` (skills/secrets/runtime info,
   per-conversation) — a prompt-caching-driven separation that is also,
   incidentally, a context-boundary discipline: static vs. situational
   information are never conflated.
5. **Tools:** `tools: list[Tool]` plus `include_default_tools` (e.g.
   `FinishTool`, `ThinkTool` — even "I'm done" is a tool call, not a
   special-cased text pattern), `filter_tools_regex` for narrowing, MCP
   tools addable at runtime via `add_runtime_tools`.
6. **Autonomy bounding:** `tool_concurrency_limit`, `filter_tools_regex`,
   `security_policy_filename`, and the workspace/sandbox boundary itself;
   an optional `critic` (`CriticBase`) for real-time action evaluation
   before/after execution.
7. **Failures:** `condenser` (`CondenserBase`) compresses history once it
   grows too large rather than erroring or silently truncating; the
   event-sourced design means a crashed run can be replayed/resumed instead
   of restarted.
8. **Success:** A `FinishTool`-style explicit action, optionally checked by
   a `critic` — but nothing external to the SDK is required to sign off.
9. **Borrow:** static-vs-dynamic context separation, "finishing is a tool
   call" (makes completion an observable, loggable event rather than a
   free-text pattern-match), the additive-only `verify()` compatibility
   rule for resuming agents, workspace-as-swappable-implementation.
10. **Unnecessary to reimplement:** event-sourced conversation state,
    condensing/summarization of long histories, the tool-calling
    ReAct-style loop, sandboxed workspace execution — again, the same class
    of harness machinery OpenCode already provides.

## Open SWE (langchain-ai/open-swe)

1. **Deterministic:** Which of five separate LangGraph graphs runs for a
   given trigger (Agent vs. Reviewer vs. Analyzer vs. Chat vs. Scheduler),
   the plan-approval gate, the CI/workflow-file human-approval carve-out,
   and the choice of sandbox provider per task.
2. **Autonomous:** Inside the Agent graph, the Planner and Programmer nodes
   — investigating a repo, proposing a plan, writing code, running
   tests/linters, iterating on failures — run with real latitude inside
   their sandbox.
3. **State:** A LangGraph thread per task, with the sandbox itself tied to
   that thread id so a workspace persists and can be resumed across turns
   instead of being torn down and rebuilt.
4. **Context:** Each of the five graphs is its own isolated execution —
   the Reviewer, for instance, gets read-only access and its own context,
   not the Agent's full working history — an architecture-level example of
   "verification must not share the worker's context."
5. **Tools:** Sandbox-scoped shell/file tools inside the Agent graph;
   GitHub API access for PR creation/comments at the orchestration level,
   not inside the sandboxed worker.
6. **Autonomy bounding:** Human plan-approval (or explicit
   auto-approval) before the Programmer acts, a hard human-approval
   requirement specifically for CI/workflow-file edits regardless of
   auto-approval settings elsewhere, and sandbox isolation per task.
7. **Failures:** The Programmer iterates on its own test/lint failures
   inside the sandbox; if the Reviewer's quality gate still fails, the
   graph loops back to the Planner rather than force-merging or giving up
   after one attempt.
8. **Success:** Explicitly **not** the same agent that wrote the code — a
   separate Reviewer graph, plus objective gates (tests, lint), decide
   whether a PR is deliverable. This is the clearest real-world precedent
   for "independent verification" in the whole survey.
9. **Borrow directly:** verifier-as-a-separate-process (this repo's
   `verifier.py` and examples `05`/`06` mirror "Reviewer graph, not
   Programmer self-report"), the named human-approval carve-out for
   high-blast-radius actions (CI/workflow edits) even inside an otherwise
   autonomous flow, sandbox-per-thread for resumable isolated workspaces.
10. **Unnecessary to reimplement:** the Planner/Programmer investigate-
    write-test-iterate loop and GitHub PR plumbing — exactly the shape of
    thing OpenCode already does locally; Open SWE is a strong reference for
    *how to wrap* that loop with review and approval, not for the loop
    itself.

---

## Cross-cutting synthesis

| Question | LangGraph | OpenCode | Deep Agents | OpenHands SDK | Open SWE |
|---|---|---|---|---|---|
| Deterministic core | graph topology | client/server + permissions | middleware composition | Workspace/EventStream plumbing | which graph runs, approval gates |
| Autonomous core | (none — must be added) | internal coding loop | one ReAct loop, high recursion ceiling | `Agent.step()` loop | Planner/Programmer in-sandbox |
| State | typed graph state + reducers | SQLite session | graph state + virtual FS | event-sourced stream | thread-scoped sandbox |
| Context crossing | up to the author | opaque session, exit code + events out | isolated vs. forked subagent state | static vs. dynamic split | separate graph per role |
| Bounding | recursion_limit, interrupt() | fail-closed permissions, agent tiers | soft (high ceiling + isolation) | concurrency/regex/security-policy limits | approval gates, sandbox scope |
| Who verifies | routing function you write | not its job (returns exit code) | not built in | optional critic | separate Reviewer graph |

The recurring pattern worth naming: **every mature project either has no
opinion on verification (LangGraph, OpenCode) or treats it as optional and
internal (Deep Agents' none, OpenHands' opt-in critic)** — except Open SWE,
which is the one project that puts verification in a structurally separate
place from the agent that did the work. That's the single idea this
repository borrows most directly, combined with Deep Agents' isolated-
subagent-context mechanism and OpenCode's fail-closed headless invocation.

## What this repository deliberately does NOT rebuild

- A ReAct/tool-calling agent loop (OpenCode, Deep Agents, and the OpenHands
  SDK all already have one; example `02` builds a *toy* one purely to make
  the concept legible, then never uses it again).
- A shell/coding agent, patch engine, or traceback parser — OpenCode's job.
- A context compressor/condenser — OpenHands' `condenser` and Deep Agents'
  `SummarizationMiddleware` both already exist; this repo's context
  isolation is about *boundaries*, not *compression*.
- A general permission/sandboxing framework — disposable OS-level
  workspaces (temp dirs / containers) plus OpenCode's own permission tiers
  cover this without a bespoke system.
