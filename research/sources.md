# Sources

Primary sources consulted for this repository's research phase. All claims in
`project_comparison.md` and the `docs/` folder trace back to something in this
list — a source file, a docstring, a CLI flag, a config default — rather than
to marketing copy. Where a project's docs site was unreachable from this
environment's network, the corresponding GitHub source was used instead;
that substitution is noted inline.

## LangGraph

- Repo: https://github.com/langchain-ai/langgraph
- `libs/langgraph/langgraph/types.py` — `Command` (the `goto` / `update` /
  `resume` state-transition object) and `Interrupt` / `interrupt()` (the
  human-in-the-loop pause-and-resume primitive). Fetched directly.
- `StateGraph` / `add_node` / `add_conditional_edges` / `.compile(checkpointer=...)`
  — the core graph-definition API (well-documented, used directly in every
  example in this repo rather than re-derived from source).
- Checkpointer implementations (`MemorySaver`, `SqliteSaver`, `PostgresSaver`)
  — durable execution / retry-from-checkpoint semantics.

## OpenCode (sst/opencode)

- Repo: https://github.com/sst/opencode
- Note: an unrelated, now-archived project also named "opencode"
  (`opencode-ai/opencode`, a Go/Bubble Tea TUI, archived 2025-09-18, successor
  renamed `crush`) shows up in search results. This repository is about the
  actively developed **sst/opencode** (TypeScript, client/server, Bun-based
  monorepo under `packages/`).
- `packages/opencode/src/cli/cmd/run.ts` (fetched via raw.githubusercontent.com)
  — the non-interactive entry point: flags (`--message`, `--model/-m`,
  `--format`, `--continue/-c`, `--session/-s`, `--fork`, `--attach`), session
  resolution, permission rules that auto-deny `question`/`plan_enter`/`plan_exit`
  prompts when running headless, event-stream subscription for progress, and
  `process.exitCode = 1` on failure.
- `opencode.ai/docs/cli` — blocked by this environment's egress proxy; CLI
  behavior above was confirmed from `run.ts` source instead.
- Built-in agent permission tiers referenced in the repo README: `build`
  (full file/bash access), `plan` (read-only, bash gated behind a permission
  prompt), `general` (a subagent reachable via `@general` for open-ended
  search) — an in-product example of "wide observation, narrow mutation."
- **Official Python client:** `opencode-ai` on PyPI
  (https://github.com/sst/opencode-sdk-python), generated from
  `packages/sdk/openapi.json` by Stainless (the same generator family
  Anthropic's own SDKs use). Installed and introspected directly in this
  environment (`pip install --pre opencode-ai`) rather than taken from
  docs, since `opencode.ai` itself is unreachable here: `Opencode(base_url=...)`
  defaults to `http://localhost:54321` or `OPENCODE_BASE_URL`, and talks to
  an already-running `opencode serve` — it does not start one itself.
  Relevant methods verified by inspecting the installed package's
  signatures and Pydantic models: `client.session.create()` (a fresh
  `Session` with its own `id`), `client.session.chat(id, provider_id=,
  model_id=, parts=[{"type": "text", "text": ...}])` (blocks until the
  assistant's turn completes, requires an explicit provider/model — no
  default), and `client.session.messages(id)` (the full
  `[{info: Message, parts: [Part, ...]}, ...]` transcript, used the same
  way `run.ts`'s stdout was: written to disk, never inlined into a
  result — see `docs/context_management.md`). `src/opencode_adapter.py`'s
  `SDKBackend` is built directly against these introspected shapes,
  cross-checked with mocks against the real `opencode_ai.types` classes
  (no live `opencode serve` was available in this environment to test
  against end-to-end).
  An earlier version of this research incorrectly stated no Python SDK
  existed; corrected here.

## Deep Agents (langchain-ai/deepagents)

- Repo: https://github.com/langchain-ai/deepagents
- `libs/deepagents/deepagents/graph.py` (fetched via raw.githubusercontent.com)
  — `create_deep_agent(...)`, which wraps LangChain's `create_agent` with a
  fixed middleware stack (filesystem, subagents, summarization, malformed
  tool-call patching, prompt caching, memory, human-in-the-loop) and a
  default `recursion_limit` of **9,999** — i.e., deliberately almost
  unbounded, left to the model and its tools to stop itself.
- `libs/deepagents/deepagents/middleware/subagents.py` (fetched directly) —
  `SubAgent` spec (`name`, `description`, `model`, `tools`, `system_prompt`,
  `mode: "isolated" | "fork"`), `SubAgentMiddleware` which adds a `task` tool;
  the `task(description, subagent_type, runtime)` implementation builds a
  **fresh** state for `"isolated"` subagents (`messages=[HumanMessage(description)]`,
  parent's message history excluded) and only merges back a single
  `ToolMessage` — the concrete mechanism this repo's context-isolation
  example is modeled on.
- `libs/deepagents/deepagents/middleware/filesystem.py` (fetched directly) —
  `BackendProtocol` with `StateBackend` (files live only in graph state,
  never touch disk, backed by a delta-channel with periodic snapshots),
  `FilesystemBackend` (real disk), `CompositeBackend` (path-prefix routing
  between backends), and `SandboxBackendProtocol` (adds an `execute` tool
  when the backend can run shell commands). Same file also lists the
  filesystem tool surface: `ls`, `read_file`, `write_file`, `edit_file`,
  `delete`, `glob`, `grep`.
- `libs/deepagents/deepagents/middleware/planning.py` — a `write_todos` tool
  giving the agent (and an observer) a visible, mutable plan without making
  plan structure part of the graph's control flow.

## OpenHands Software Agent SDK (OpenHands/software-agent-sdk)

- Repo: https://github.com/OpenHands/software-agent-sdk
  (formerly `All-Hands-AI/OpenHands`; V1 SDK split out of the monolithic V0
  agent into four packages: `openhands.sdk`, `tools`, `workspaces`, `server`).
- `openhands-sdk/openhands/sdk/agent/base.py` (fetched directly) —
  `AgentBase`: `llm`, `tools`, `include_default_tools` (built-ins such as
  `FinishTool`, `ThinkTool` — note that *finishing* is itself a tool call,
  not free-form text), `filter_tools_regex`, `tool_concurrency_limit`,
  `system_prompt` / `system_prompt_filename`, `agent_context` (skills /
  secrets / memory), `security_policy_filename`, `condenser` (a
  `CondenserBase` for compressing history when it grows too large), `critic`
  (an experimental real-time action evaluator). Core loop method
  `step(conversation, on_event, on_token)` explicitly mutates state in place
  and is left abstract for subclasses. `verify(persisted)` enforces that a
  resumed agent config may only **add** tools relative to a persisted one,
  never remove — a compatibility/safety constraint on resumption.
- `docs.openhands.dev/sdk` — blocked by this environment's egress proxy;
  package/architecture description corroborated via search-result summaries
  and the `base.py` source above.
- Older V0 architecture write-ups (via search) describe the underlying
  event-sourced design that the SDK still follows conceptually: `EventStream`
  as a typed pub/sub bus, `AgentController` turning LLM output into
  `Action`s, a `Runtime` executing actions inside a sandbox and returning
  `Observation`s back onto the stream.

## Open SWE (langchain-ai/open-swe)

- Repo: https://github.com/langchain-ai/open-swe
- Architecture (via repo README/search, LangGraph-based): five separate
  graph entrypoints — **Agent** (plans, implements, validates, delivers),
  **Reviewer** (read-only PR analysis, a quality gate distinct from the
  agent that wrote the code), **Analyzer** (learns a repo's review style
  over time), **Chat** (answers questions, never mutates), **Scheduler**
  (dispatches recurring/CI-triggered runs).
- Sandbox providers are pluggable: LangSmith (default), Modal, Daytona,
  Runloop, E2B, or local — each task's workspace is tied to a LangGraph
  thread id so a sandbox can be resumed across turns instead of rebuilt.
- Human-in-the-loop gate: plans require approval (or explicit
  auto-approval) before the Programmer executes them; CI/workflow-file
  changes specifically require human approval even when the rest of a run
  is unattended — a narrow, named exception carved out of an otherwise
  autonomous loop.
- Completion criteria are external to the agent that wrote the code: tests
  passing, lint passing, and a separate Reviewer graph's approval gate the
  PR before delivery; a "needs follow-up" result loops back to the Planner
  rather than being force-closed.

## How these sources map onto this repository

Every non-trivial design choice in `docs/` and `src/` cites one of the
patterns above by name (e.g. "isolated-mode subagent state" ->
`context_management.md`, "SandboxBackendProtocol" -> `sandboxing.md`,
"Reviewer as a separate graph" -> `verification.md`). See
`project_comparison.md` for the structured, per-project breakdown.
