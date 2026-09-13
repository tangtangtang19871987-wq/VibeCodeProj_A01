# Any Streamable HTTP MCP client

The daemon speaks standard MCP Streamable HTTP
(`2025-03-26`+ transport spec) at:

```
http://127.0.0.1:4317/mcp
```

- Method: `POST` for every request (the daemon runs stateless per
  request — see ADR 0005 — so no `GET`/SSE stream or session-ID header is
  required).
- No authentication by default (v1 is single-user, loopback-only —
  Section 18.1/18.2). Do not expose this port beyond localhost.

Using the official TypeScript SDK directly:

```ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const client = new Client({ name: "my-harness", version: "0.0.1" });
await client.connect(new StreamableHTTPClientTransport(new URL("http://127.0.0.1:4317/mcp")));

const { tools } = await client.listTools();

const session = await client.callTool({
  name: "memory_session",
  arguments: { action: "open", project: "my-project", harness: "my-harness" },
});
```

See `apps/daemon/src/mcp.e2e.test.ts` in this repo for a complete worked
example covering all five tools, including a two-client scenario proving
memory is genuinely shared across sessions.
