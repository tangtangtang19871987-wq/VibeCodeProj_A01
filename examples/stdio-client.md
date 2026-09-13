# A harness that only supports local stdio MCP servers

Use the forwarding shim (ADR 0005) — it relays JSON-RPC messages between
your harness's stdio and the daemon's `/mcp` endpoint, with no tool logic
of its own:

```bash
node apps/daemon/dist/cli.js mcp-stdio
```

(or `pnpm --filter @lams/daemon run cli mcp-stdio` in development).

The daemon must already be running (`pnpm dev`, or
`node apps/daemon/dist/server.js`) — the shim is a client of it, not a
replacement for it.

Configure your harness to launch this as a local MCP server, e.g. in a
Claude Desktop-style `mcpServers` config:

```json
{
  "mcpServers": {
    "lams": {
      "command": "node",
      "args": ["/absolute/path/to/apps/daemon/dist/cli.js", "mcp-stdio"]
    }
  }
}
```

If your `LAMS_PORT`/`LAMS_HOST` differ from the defaults, set the same
environment variables when launching the shim so it targets the right
daemon.
