# ADR 0007: UI serving strategy

Status: Accepted

## Context

Section 25 item 2 leaves open whether production serves a prebuilt React
bundle from Fastify or runs a separate UI process.

## Decision

- Development: `apps/studio` runs its own Vite dev server (fast HMR); the
  daemon's Fastify instance proxies `/api/*` and `/mcp` are called directly
  by the UI's dev server via Vite's proxy config, so a developer only visits
  one URL.
- Production (`pnpm build` then `lams start`): Vite builds `apps/studio` to
  static assets, and the daemon serves them directly via `@fastify/static`
  from the same Fastify instance and port as the REST API and MCP endpoint.

## Consequences

There is exactly one process and one port to run and firewall in
production, matching Section 8.1's "one long-lived daemon process." In
development, two processes run (daemon + Vite) started together by
`pnpm dev` (ADR 0004), giving fast UI iteration without rebuilding.

## Alternatives considered

Always running the UI as a separate process (even in production): rejected
because it complicates the "one command, one daemon" story in Section 21's
Milestone 0 acceptance criteria and Section 23's definition of done.
