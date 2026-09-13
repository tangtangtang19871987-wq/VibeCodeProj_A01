# Claude Code

Claude Code's CLI can add a remote (Streamable HTTP) MCP server directly:

```bash
claude mcp add --transport http lams http://127.0.0.1:4317/mcp
```

This writes an entry to Claude Code's MCP config (project-local `.mcp.json`
by default, or pass `--scope user`/`--scope local` to change where it's
stored). Verify it's connected:

```bash
claude mcp list
```

You should see `lams` listed as connected, exposing `memory_session`,
`memory_recall`, `memory_get`, `memory_remember`, and `memory_feedback`.

Equivalently, you can hand-edit `.mcp.json`:

```json
{
  "mcpServers": {
    "lams": {
      "type": "http",
      "url": "http://127.0.0.1:4317/mcp"
    }
  }
}
```

Start the LAMS daemon (`pnpm dev`, or `node apps/daemon/dist/server.js`
after `pnpm build`) before starting a Claude Code session that needs it.
