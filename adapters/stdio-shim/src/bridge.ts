import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

/**
 * ADR 0005: a thin, business-logic-free forwarder between an MCP stdio
 * client (the harness) and the daemon's Streamable HTTP endpoint. It knows
 * nothing about sessions, memories, or tool schemas — it only relays
 * JSON-RPC messages in both directions, since MCP's `Transport` interface
 * is itself protocol-agnostic (a `JSONRPCMessage` in, a `JSONRPCMessage`
 * out). All five tools are defined once, in `@lams/mcp-adapter`, and
 * served identically whether a harness connects here or directly over
 * HTTP.
 */
export async function runStdioShim(daemonMcpUrl: string): Promise<void> {
  const stdio = new StdioServerTransport();
  const http = new StreamableHTTPClientTransport(new URL(daemonMcpUrl));

  stdio.onmessage = (message) => {
    void http.send(message).catch((err) => stdio.onerror?.(err as Error));
  };
  http.onmessage = (message) => {
    void stdio.send(message).catch((err) => stdio.onerror?.(err as Error));
  };

  let closed = false;
  const shutdown = () => {
    if (closed) return;
    closed = true;
    void stdio.close();
    void http.close();
  };
  stdio.onclose = shutdown;
  http.onclose = shutdown;
  stdio.onerror = (err) => console.error("[lams stdio-shim] stdio error:", err);
  http.onerror = (err) => console.error("[lams stdio-shim] http error:", err);

  await http.start();
  await stdio.start();
}
