# ADR 0005: MCP transport

Status: Accepted

## Context

Section 12.1 requires Streamable HTTP MCP as the primary transport, with an
optional stdio shim for harnesses that only support local stdio servers.
Section 25 item 3 leaves the exact SDK transport open.

## Decision

Use the official `@modelcontextprotocol/sdk` TypeScript SDK's
`StreamableHTTPServerTransport`, mounted directly on the daemon's Fastify
instance at `/mcp`, bound to `127.0.0.1` by default. A separate thin process
(`apps/daemon/src/cli.ts mcp-stdio`, under `adapters/stdio-shim`) speaks MCP
stdio to the harness and forwards every call to the daemon's `/mcp` endpoint
over loopback HTTP — it contains no business logic, per Section 8.1.

## Consequences

The five MCP tools (Section 12.2) are defined once, against the application
service layer, and served identically over HTTP or stdio. Adding a new
harness that only speaks stdio never requires touching tool logic.

## Alternatives considered

Embedding MCP tool logic directly in each harness adapter: rejected because
it would duplicate business logic per harness and violate Section 8.2's
requirement that MCP be a thin adapter over the same application services
used by REST.
