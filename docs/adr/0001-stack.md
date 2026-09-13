# ADR 0001: Implementation stack

Status: Accepted

## Context

The PRD (Section 8.3) recommends a TypeScript-first monorepo to avoid a Python
dependency for the MVP and to share types across the daemon, MCP adapter, and
UI. We need to fix the stack before Milestone 0 work can proceed.

## Decision

- Node.js 22+, TypeScript in strict mode, pnpm workspaces.
- Fastify for the daemon HTTP API.
- Official `@modelcontextprotocol/sdk` for MCP transport and tool definitions.
- `better-sqlite3` as the SQLite driver (synchronous, simplest correctness
  model for a single-process local daemon; see ADR 0002 for the query layer
  built on top of it).
- React + Vite for the web UI, TanStack Query for server state.
- Vitest for unit/integration tests, Playwright for UI acceptance tests.
- Zod for runtime contract validation shared between REST/MCP boundaries.
- Pino for structured operational logging (kept separate from the product
  trace tables per Section 16.1).

## Consequences

The MVP has no Python or external service dependency. All packages share one
TypeScript type system, which lets `packages/domain` types flow unmodified
into the REST layer, the MCP tool schemas, and the UI.

## Alternatives considered

A Python/FastAPI daemon was rejected because it would require a second
language for the eventual React UI boundary and duplicate type definitions
across a network boundary that provides no benefit for a single local
process.
