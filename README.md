# LangGraph-Orchestrated Agent with an OpenCode Kernel

A research-oriented teaching repository. It demonstrates one specific,
narrow architecture — not a new agent framework, not a replacement for
OpenCode:

> **LangGraph is the main deterministic orchestrator. OpenCode is a bounded
> autonomous agent kernel inside selected nodes.**

```text
                LangGraph
        deterministic control plane
                    |
        ---------------------------
        |            |            |
 deterministic    LLM node    OpenCode node
 domain logic                  autonomous kernel
        |                         |
        ---------------------------
                    |
              verification
                    |
              next graph step
```

LangGraph always decides: what task is assigned, what context is provided,
what workspace is available, what resources/permissions are allowed, retry
and budget limits, what counts as success, and what happens next. Inside
that boundary, OpenCode is free to investigate, edit, run, fail, and retry
using its own internal loop — a loop this repository deliberately does not
rebuild.

## Central research question

Can strong domain-specific agents be built with much less custom
agent-harness engineering by combining LangGraph orchestration + an
OpenCode autonomous kernel + deterministic verification, while keeping
OpenCode's investigation/debugging/self-repair ability intact and never
reimplementing it? `docs/` and `research/` work through the answer in
detail; `examples/` makes it concrete and runnable.

## Start here

New to this repository? Follow `LEARNING_PATH.md` in order — it's a guided
path through `research/` -> `docs/` -> `examples/00` through `09`, with a
specific question to answer at each stop.

## Layout

```text
README.md              — you are here
LEARNING_PATH.md        — the guided path through this repo

research/
    sources.md               — primary sources: repos, files, functions cited
    project_comparison.md    — LangGraph / OpenCode / Deep Agents / OpenHands SDK / Open SWE,
                                compared on 10 fixed questions

docs/
    architecture.md          — the shape of the system and why
    context_management.md    — what's allowed to cross the LangGraph<->OpenCode boundary
    bounded_autonomy.md       — the 6 things LangGraph never delegates
    verification.md          — why OpenCode never gets to define its own success
    sandboxing.md            — disposable workspaces over custom permission systems
    failure_recovery.md      — task-level retries vs. OpenCode's own internal retries
    engineering_tradeoffs.md — where this works, where it doesn't, and honest limits

examples/
    00_basic_langgraph                          — pure deterministic graph
    01_llm_inside_graph                         — one LLM call, no loop
    02_minimal_agent_loop                       — a toy LLM->tool->observation loop (educational only)
    03_opencode_as_node                         — OpenCode as a single LangGraph node
    04_autonomous_coding_worker                 — OpenCode investigates/edits/repairs a broken project
    05_external_verifier                        — a deterministic verifier catches an over-confident claim
    06_verifier_feedback_loop                   — failed verification retries with feedback
    07_context_isolation                        — OpenCode's internal chatter never bloats graph state
    08_disposable_workspace                     — create -> work -> verify -> promote or discard
    09_domain_workflow_with_agent_escape_hatch  — the capstone: autonomy as one bounded branch of a real workflow

src/
    contracts.py          — AgentTask / AgentResult / VerificationResult
    opencode_adapter.py    — the only place that invokes OpenCode (SDK, CLI, or an honest fake backend)
    verifier.py            — deterministic, OpenCode-independent success checks
```

## Running the examples

```bash
pip install -r requirements.txt
python examples/00_basic_langgraph/main.py
# ...through...
python examples/09_domain_workflow_with_agent_escape_hatch/main.py
```

Every example runs out of the box with no external services, API keys, or
real OpenCode installation:

- `examples/01` falls back to a clearly-labeled deterministic stub if
  `ANTHROPIC_API_KEY` isn't set.
- `examples/03` onward default to `src/opencode_adapter.py`'s `FakeBackend`
  — an explicit, honest test double (see its docstring), not a
  reimplementation of OpenCode. To run any of them against the real thing,
  with zero code changes — that portability is the entire point of
  `src/opencode_adapter.py` existing:
  - `OPENCODE_ADAPTER_BACKEND=sdk` — talks to an already-running
    `opencode serve` through the official `opencode-ai` Python client
    (`pip install --pre opencode-ai`; also set
    `AgentTask.context['provider_id']`/`['model_id']` or
    `OPENCODE_PROVIDER`/`OPENCODE_MODEL`). This is the preferred backend
    when a server is available — structured responses instead of parsed
    CLI text — and `OPENCODE_ADAPTER_BACKEND=auto` (the default) already
    picks it automatically whenever a server is reachable.
  - `OPENCODE_ADAPTER_BACKEND=cli` — shells out to the `opencode` CLI
    (needs the binary on `PATH` and a model configured), no server
    required.

## What this repository is not

Not another general-purpose agent framework, not a replacement for
OpenCode, not a place where Deep Agents/OpenHands/Open SWE get rebuilt.
Those projects (and others like them) are studied in `research/` for their
engineering ideas, several of which are borrowed directly — see
`research/project_comparison.md`'s "cross-cutting synthesis" and "what this
repository deliberately does NOT rebuild" sections.
