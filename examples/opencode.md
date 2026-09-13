# OpenCode

OpenCode reads MCP server definitions from its config file (`opencode.json`
in your project root, or the global `~/.config/opencode/opencode.json`).
Add a `remote`-type entry pointing at the daemon's Streamable HTTP
endpoint:

```json
{
  "mcp": {
    "lams": {
      "type": "remote",
      "url": "http://127.0.0.1:4317/mcp",
      "enabled": true
    }
  }
}
```

Start the LAMS daemon first (`pnpm dev` from this repo, or
`node apps/daemon/dist/server.js` after `pnpm build`), then start
OpenCode — it should list `memory_session`, `memory_recall`, `memory_get`,
`memory_remember`, and `memory_feedback` among its available tools.

> OpenCode's exact MCP configuration schema has changed across releases;
> check `opencode --help` or OpenCode's own MCP documentation for your
> installed version if this doesn't match. The important part is pointing
> a Streamable HTTP MCP client at `http://127.0.0.1:4317/mcp`.

## Suggested first prompt

Once connected, a reasonable smoke test inside OpenCode:

```
Open a LAMS session for this project, then recall anything relevant to
"database migrations", and tell me what you found.
```
