import { createMcpServer, type McpToolContext } from "@lams/mcp-adapter";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import type { FastifyInstance } from "fastify";
import type { AppContext } from "../context.js";

function toolContext(ctx: AppContext): McpToolContext {
  return {
    projectService: ctx.projectService,
    sessionService: ctx.sessionService,
    memoryService: ctx.memoryService,
    retrievalService: ctx.retrievalService,
    feedbackService: ctx.feedbackService,
  };
}

/**
 * ADR 0005: Streamable HTTP MCP mounted directly on the daemon's Fastify
 * instance. Stateless per PRD's session model (Section 9.5's AgentSession
 * is our own domain concept, tracked via memory_session — MCP transport-
 * level sessions add nothing here), so each request gets a fresh
 * server+transport pair, matching the SDK's own stateless example.
 */
export async function mcpRoutes(app: FastifyInstance, ctx: AppContext) {
  app.post("/mcp", async (req, reply) => {
    const server = createMcpServer(toolContext(ctx));
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });

    reply.hijack();
    try {
      await server.connect(transport);
      await transport.handleRequest(req.raw, reply.raw, req.body);
    } catch (err) {
      req.log.error({ err }, "Error handling MCP request");
      if (!reply.raw.headersSent) {
        reply.raw.writeHead(500, { "Content-Type": "application/json" });
        reply.raw.end(
          JSON.stringify({
            jsonrpc: "2.0",
            error: { code: -32603, message: "Internal server error" },
            id: null,
          }),
        );
      }
    } finally {
      reply.raw.on("close", () => {
        transport.close();
        server.close();
      });
    }
  });
}
