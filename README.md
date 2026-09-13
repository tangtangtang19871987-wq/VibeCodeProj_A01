# Local Agent Memory Studio (LAMS)

A local-first, inspectable memory daemon for AI coding and engineering
agents — with human review, full retrieval traces, and (from Milestone 3
onward) a context-efficient MCP interface for agent harnesses like
OpenCode, GitHub Copilot CLI, and Claude Code.

See [`local-agent-memory-studio-prd.md`](./local-agent-memory-studio-prd.md)
(if present in your copy) or the task description this repository was built
from for the full product requirements. Architectural decisions are recorded
in [`docs/adr/`](./docs/adr/) as they are made.

This repository also contains an unrelated pre-existing Python module under
[`legacy/array-generator/`](./legacy/array-generator/); it is untouched by
LAMS development.

## Status

Milestones 0 through 3 are complete:

- **Milestone 0** — architecture skeleton: a runnable daemon with a health
  endpoint, SQLite + FTS5 storage, and a served web UI shell.
- **Milestone 1** — memory vertical slice: create/version/review/search
  memories end to end, with full provenance, review-action, and lineage
  history, through both the REST API and the Memory Explorer UI.
- **Milestone 2** — sessions and retrieval observability: a deterministic
  FTS + scoring retrieval pipeline (`packages/retrieval`) with transparent
  score components and reason codes, a hard character budget, full
  recall-trace/candidate/delivery-event persistence, and a Session
  Inspector + Retrieval Inspector in the UI. Every recall — even an empty
  one — persists a complete, inspectable trace.
- **Milestone 3** — MCP daemon integration: exactly the five tools defined
  in the PRD (`memory_session`, `memory_recall`, `memory_get`,
  `memory_remember`, `memory_feedback`), served over Streamable HTTP at
  `/mcp` and, via a thin forwarding shim, over stdio for harnesses that
  need it. See [`examples/`](./examples/) for OpenCode, Claude Code, and
  generic client setup. A real MCP client test (`apps/daemon/src/mcp.e2e.test.ts`)
  proves two independent clients share one memory universe end to end.

Milestone 4 (governance/transfer: a dedicated Review Queue, lineage
navigation, export/backup/import) is next. See `docs/adr/` and the PRD's
milestone list for details.

## Requirements

- Node.js 22+
- pnpm (`corepack enable` or `npm i -g pnpm`)

No external services, API keys, or Python are required.

## Quickstart

```bash
pnpm install
pnpm dev
```

This starts the daemon (REST API + MCP endpoint, once implemented) on
`http://127.0.0.1:4317` and the Studio UI's Vite dev server on
`http://127.0.0.1:5173` (which proxies `/api` and `/mcp` to the daemon) —
open the latter in your browser.

Data is stored under an OS-appropriate application data directory by
default (see `docs/adr/0004-process-and-data-directory.md`); override with
`LAMS_DATA_DIR=/path/to/dir`.

## Production-style run

```bash
pnpm build
LAMS_DATA_DIR=./.lams-data NODE_ENV=production node apps/daemon/dist/server.js
```

This serves the prebuilt UI and the API from one process on one port.

## Repository layout

```text
apps/
  daemon/    # Fastify HTTP API, MCP endpoint (later), CLI, process lifecycle
  studio/    # React + Vite web UI
packages/
  domain/         # Entities, value objects, and pure policy functions — no framework deps
  retrieval/       # Pure scoring/ranking/eligibility functions for recall — depends only on domain
  application/     # Use cases and port interfaces — depends only on domain (+ retrieval's pure functions)
  storage-sqlite/  # SQLite schema, migrations, FTS5, repository implementations
  mcp-adapter/     # The five MCP tools, mapped onto application services — no storage/Fastify/React deps
adapters/
  stdio-shim/      # Business-logic-free stdio <-> Streamable HTTP forwarder (ADR 0005)
docs/
  adr/       # Architecture decision records
legacy/
  array-generator/  # Pre-existing, unrelated Python module (untouched)
```

`packages/domain` and `packages/application` must never import Fastify,
React, the MCP SDK, or SQLite details directly — this is enforced by
`pnpm arch:check` (dependency-cruiser) in CI.

## Development commands

| Command | What it does |
|---|---|
| `pnpm dev` | Run daemon + UI together for local development |
| `pnpm build` | Build all packages/apps (including the UI bundle) |
| `pnpm typecheck` | TypeScript strict-mode check across the workspace |
| `pnpm lint` | ESLint |
| `pnpm test` | Vitest unit/integration tests (no external services required) |
| `pnpm test:e2e` | Playwright UI smoke tests against a real built daemon |
| `pnpm arch:check` | Enforce the architectural dependency rules above |
| `pnpm --filter @lams/daemon run seed` | Populate demo data (the PRD Section 27 scenario) into the configured data dir |

## Design principles (short version)

- Deterministic retrieval before anything LLM- or embedding-based.
- Every memory has provenance, versions, and a review history — agent
  output is never silently authoritative.
- Recall is fully traced: what was eligible, scored, selected, returned —
  and, separately and honestly labeled, what a harness *reported* was
  injected or used.
- One local daemon, one SQLite database, no cloud dependency.

See `docs/adr/` for the reasoning behind specific technical choices.
