import { MEMORY_KINDS } from "@lams/domain";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { McpToolContext } from "../context.js";
import { errorResult, textResult } from "../result.js";

const DEFAULT_MAX_ITEMS = 5;
const DEFAULT_MAX_CHARS = 2000;

const inputShape = {
  session_id: z.string().describe("A session_id from memory_session."),
  query: z.string().min(1).max(1000),
  intent: z.string().max(500).optional().describe("Why this recall is being made, for the trace."),
  kinds: z.array(z.enum(MEMORY_KINDS)).max(6).optional(),
  max_items: z.number().int().min(1).max(10).optional().describe(`Default ${DEFAULT_MAX_ITEMS}.`),
  max_chars: z.number().int().min(200).max(8000).optional().describe(`Default ${DEFAULT_MAX_CHARS}.`),
  include_drafts: z.boolean().optional().describe("Default false — draft memories are excluded unless requested."),
};

export function registerMemoryRecall(server: McpServer, ctx: McpToolContext) {
  server.registerTool(
    "memory_recall",
    {
      title: "Recall a bounded, task-relevant memory packet",
      description:
        "Creates persistent data: every call persists a full recall trace and candidate list (visible in the Studio UI's Retrieval Inspector), even when nothing is returned. The response itself only contains the selected memories, compact reasons, and a trace_id — use memory_get for full detail on a specific memory.",
      inputSchema: inputShape,
    },
    async (args) => {
      try {
        const { packet } = await ctx.retrievalService.recall({
          sessionId: args.session_id,
          query: args.query,
          intent: args.intent,
          kinds: args.kinds,
          maxItems: args.max_items ?? DEFAULT_MAX_ITEMS,
          maxChars: args.max_chars ?? DEFAULT_MAX_CHARS,
          includeDrafts: args.include_drafts ?? false,
        });
        return textResult(packet);
      } catch (err) {
        return errorResult("NOT_FOUND", (err as Error).message);
      }
    },
  );
}
