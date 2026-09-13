import { MEMORY_KINDS } from "@lams/domain";
import type { FastifyInstance } from "fastify";
import { z } from "zod";
import type { AppContext } from "../context.js";

const recallSchema = z.object({
  sessionId: z.string().min(1),
  query: z.string().min(1).max(1000),
  intent: z.string().max(500).optional(),
  kinds: z.array(z.enum(MEMORY_KINDS)).max(10).optional(),
  maxItems: z.number().int().min(1).max(20).optional(),
  maxChars: z.number().int().min(100).max(20_000).optional(),
  includeDrafts: z.boolean().optional(),
});

export async function recallRoutes(app: FastifyInstance, ctx: AppContext) {
  app.post("/api/v1/recalls", async (req, reply) => {
    const parsed = recallSchema.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400);
      return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
    }
    try {
      const { packet, trace } = await ctx.retrievalService.recall(parsed.data);
      reply.code(201);
      return { packet, trace };
    } catch (err) {
      reply.code(404);
      return { error: { code: "NOT_FOUND", message: (err as Error).message } };
    }
  });

  app.get<{ Params: { id: string } }>("/api/v1/recalls/:id", async (req, reply) => {
    const trace = await ctx.recallRepo.getTrace(req.params.id);
    if (!trace) {
      reply.code(404);
      return { error: { code: "NOT_FOUND", message: `Recall trace not found: ${req.params.id}` } };
    }
    return { trace };
  });

  app.get<{ Params: { id: string } }>(
    "/api/v1/recalls/:id/candidates",
    async (req, reply) => {
      const trace = await ctx.recallRepo.getTrace(req.params.id);
      if (!trace) {
        reply.code(404);
        return { error: { code: "NOT_FOUND", message: `Recall trace not found: ${req.params.id}` } };
      }
      const candidates = await ctx.recallRepo.listCandidates(req.params.id);
      // Enrich each candidate with the memory-version snapshot it pointed
      // to at query time (Section 15.4 — versions are immutable, so this
      // always reflects what the recall actually saw, even if the memory
      // has since been edited further).
      const enriched = await Promise.all(
        candidates.map(async (c) => {
          const version = await ctx.memoryService.getVersion(c.memoryVersionId);
          return { ...c, title: version?.title, content: version?.content };
        }),
      );
      const deliveryEvents = await ctx.deliveryRepo.listForTrace(req.params.id);
      return { trace, candidates: enriched, deliveryEvents };
    },
  );
}
