import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { McpToolContext } from "../context.js";
import { textResult } from "../result.js";

const MAX_IDS = 10;

const inputShape = {
  session_id: z.string(),
  memory_ids: z.array(z.string()).min(1).max(MAX_IDS),
  include_provenance: z.boolean().optional(),
  include_history: z.boolean().optional(),
};

export function registerMemoryGet(server: McpServer, ctx: McpToolContext) {
  server.registerTool(
    "memory_get",
    {
      title: "Fetch full detail for a small, explicit list of memories",
      description:
        `Reads only — no persistent data is created. For deliberate drill-down on IDs you already have (e.g. from memory_recall), not broad search; capped at ${MAX_IDS} IDs per call.`,
      inputSchema: inputShape,
    },
    async (args) => {
      const items = [];
      for (const id of args.memory_ids) {
        const detail = await ctx.memoryService.getDetail(id);
        if (!detail) {
          items.push({ id, error: { code: "NOT_FOUND", message: `Memory not found: ${id}` } });
          continue;
        }
        items.push({
          id: detail.memory.id,
          kind: detail.memory.kind,
          status: detail.memory.status,
          scope: detail.memory.scope,
          title: detail.currentVersion.title,
          content: detail.currentVersion.content,
          tags: detail.currentVersion.tags,
          updatedAt: detail.memory.updatedAt,
          ...(args.include_provenance
            ? {
                provenance: detail.provenance.map((p) => ({
                  sourceType: p.sourceType,
                  sourceRef: p.sourceRef,
                  excerpt: p.excerpt,
                })),
              }
            : {}),
          ...(args.include_history
            ? {
                versionCount: detail.versions.length,
                reviewCount: detail.reviews.length,
              }
            : {}),
        });
      }
      return textResult({ memories: items });
    },
  );
}
