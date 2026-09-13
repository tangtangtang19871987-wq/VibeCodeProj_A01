import { agentActor, MEMORY_KINDS, MEMORY_RELATION_TYPES } from "@lams/domain";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { McpToolContext } from "../context.js";
import { errorResult, textResult } from "../result.js";

const MAX_CANDIDATES = 5; // ADR 0012

const candidateShape = z.object({
  kind: z.enum(MEMORY_KINDS),
  title: z.string().min(1).max(300),
  content: z.string().min(1).max(20_000),
  scope: z
    .object({ level: z.enum(["global", "user", "project", "agent", "session"]), id: z.string().optional() })
    .optional()
    .describe("Defaults to the session's project scope if omitted."),
  confidence: z.number().min(0).max(1).optional(),
  tags: z.array(z.string()).max(20).optional(),
  evidence: z
    .array(
      z.object({
        sourceType: z.enum(["session_event", "tool_result", "artifact", "human", "import", "memory"]),
        sourceRef: z.string(),
        excerpt: z.string().max(2000).optional(),
      }),
    )
    .max(5)
    .optional(),
  reasonWorthRemembering: z.string().max(500).optional(),
  relation: z
    .object({
      type: z.enum(MEMORY_RELATION_TYPES),
      targetMemoryId: z.string(),
    })
    .optional(),
});

const inputShape = {
  session_id: z.string(),
  candidates: z.array(candidateShape).min(1).max(MAX_CANDIDATES),
};

export function registerMemoryRemember(server: McpServer, ctx: McpToolContext) {
  server.registerTool(
    "memory_remember",
    {
      title: "Propose one or more memory candidates from this session",
      description:
        `Creates persistent data: each candidate becomes a new draft memory (agent-created memories are always draft, per human-governance policy) plus any evidence as provenance. Up to ${MAX_CANDIDATES} candidates per call.`,
      inputSchema: inputShape,
    },
    async (args) => {
      const session = await ctx.sessionService.getById(args.session_id);
      if (!session) return errorResult("NOT_FOUND", `Session not found: ${args.session_id}`);

      const actor = agentActor(session.agentName ?? session.harness, session.agentName);
      const results = [];

      for (const candidate of args.candidates) {
        const scope = candidate.scope ?? { level: "project" as const, id: session.projectId };
        const { memory } = await ctx.memoryService.create({
          kind: candidate.kind,
          scope,
          confidence: candidate.confidence,
          createdBy: actor,
          version: {
            title: candidate.title,
            content: candidate.content,
            tags: candidate.tags,
            changeReason: candidate.reasonWorthRemembering,
          },
        });

        for (const ev of candidate.evidence ?? []) {
          await ctx.memoryService.addProvenance({
            memoryId: memory.id,
            sourceType: ev.sourceType,
            sourceRef: ev.sourceRef,
            excerpt: ev.excerpt,
            sessionId: session.id,
          });
        }

        if (candidate.relation) {
          await ctx.memoryService.addRelation({
            fromMemoryId: memory.id,
            toMemoryId: candidate.relation.targetMemoryId,
            type: candidate.relation.type,
            createdBy: actor,
          });
        }

        results.push({ memory_id: memory.id, status: memory.status });
      }

      return textResult({ candidates: results });
    },
  );
}
