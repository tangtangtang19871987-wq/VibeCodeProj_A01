# Connecting a harness to LAMS

The daemon serves MCP over Streamable HTTP at `http://127.0.0.1:4317/mcp`
(bound to loopback only, per Section 18.1). Start it first:

```bash
pnpm dev        # development, or
pnpm build && node apps/daemon/dist/server.js   # production-style
```

Then point your harness at `http://127.0.0.1:4317/mcp` using whichever of
the examples below matches it. If your harness only supports local stdio
MCP servers, use [`stdio-client.md`](./stdio-client.md) — every one of the
five tools behaves identically over either transport (ADR 0005), since the
stdio path is a thin, business-logic-free forward to the same `/mcp`
endpoint.

- [`opencode.md`](./opencode.md) — OpenCode
- [`claude-code.md`](./claude-code.md) — Claude Code
- [`stdio-client.md`](./stdio-client.md) — any client that only speaks
  local stdio MCP (via the forwarding shim)
- [`generic-streamable-http.md`](./generic-streamable-http.md) — any other
  MCP client that supports Streamable HTTP directly

## What a harness gets

Five tools, all defined once in `packages/mcp-adapter` and mapped straight
onto the same application services the REST API and Studio UI use
(Section 8.2):

| Tool | Purpose |
|---|---|
| `memory_session` | open / update / close a tracked session |
| `memory_recall` | get a bounded, ranked memory packet for a query |
| `memory_get` | fetch full detail for up to 10 memory IDs |
| `memory_remember` | propose up to 5 memory candidates (always created as drafts) |
| `memory_feedback` | report injected/useful/harmful/outcome feedback |

Every call is visible immediately in the Studio UI — recalls in the
Retrieval Inspector, sessions in the Session Inspector, remembered
candidates (as drafts) in the Memory Explorer / Review Queue.
