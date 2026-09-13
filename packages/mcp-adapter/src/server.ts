import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { McpToolContext } from "./context.js";
import { registerMemorySession } from "./tools/memory-session.js";
import { registerMemoryRecall } from "./tools/memory-recall.js";
import { registerMemoryGet } from "./tools/memory-get.js";
import { registerMemoryRemember } from "./tools/memory-remember.js";
import { registerMemoryFeedback } from "./tools/memory-feedback.js";

/**
 * PRD Section 12.2: exactly five tools, no more. Adding a sixth requires a
 * PRD/ADR change, not just a new file in tools/.
 */
export function createMcpServer(ctx: McpToolContext): McpServer {
  const server = new McpServer({ name: "local-agent-memory-studio", version: "0.1.0" });

  registerMemorySession(server, ctx);
  registerMemoryRecall(server, ctx);
  registerMemoryGet(server, ctx);
  registerMemoryRemember(server, ctx);
  registerMemoryFeedback(server, ctx);

  return server;
}
