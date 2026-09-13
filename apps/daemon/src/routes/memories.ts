import {
  InvalidStatusTransitionError,
  MEMORY_KINDS,
  MEMORY_STATUSES,
  parseScope,
  scopeToString,
  type ScopeRef,
} from "@lams/domain";
import type { FastifyInstance } from "fastify";
import { z } from "zod";
import type { AppContext } from "../context.js";
import { actorInputSchema, resolveActor } from "./actor.js";

const scopeSchema = z.object({
  level: z.enum(["global", "user", "project", "agent", "session"]),
  id: z.string().optional(),
});

const versionInputSchema = z.object({
  title: z.string().min(1).max(300),
  content: z.string().min(1).max(20_000),
  summary: z.string().max(2000).optional(),
  tags: z.array(z.string().min(1).max(60)).max(20).optional(),
  structuredData: z.record(z.unknown()).optional(),
  changeReason: z.string().max(2000).optional(),
});

const createMemorySchema = z.object({
  kind: z.enum(MEMORY_KINDS),
  scope: scopeSchema,
  status: z.enum(MEMORY_STATUSES).optional(),
  confidence: z.number().min(0).max(1).optional(),
  importance: z.number().min(0).max(1).optional(),
  validFrom: z.string().optional(),
  validUntil: z.string().optional(),
  version: versionInputSchema,
  actor: actorInputSchema,
});

const addVersionSchema = versionInputSchema.extend({ actor: actorInputSchema });

const provenanceSchema = z.object({
  sourceType: z.enum(["session_event", "tool_result", "artifact", "human", "import", "memory"]),
  sourceRef: z.string().min(1).max(500),
  memoryVersionId: z.string().optional(),
  sessionId: z.string().optional(),
  eventId: z.string().optional(),
  excerpt: z.string().max(4000).optional(),
  artifactHash: z.string().optional(),
});

const reviewActionSchema = z.object({
  action: z.enum([
    "approve",
    "reject",
    "deprecate",
    "restore",
    "supersede",
    "merge",
  ]),
  reason: z.string().max(2000).optional(),
  targetMemoryId: z.string().optional(), // required for supersede/merge
  actor: actorInputSchema,
});

const listQuerySchema = z.object({
  q: z.string().optional(),
  status: z.string().optional(), // comma-separated
  kind: z.string().optional(), // comma-separated
  scope: z.string().optional(), // "level/id"
  includeDeleted: z.coerce.boolean().optional(),
  limit: z.coerce.number().int().min(1).max(200).optional(),
  cursor: z.string().optional(),
});

function errorResponse(err: unknown) {
  if (err instanceof InvalidStatusTransitionError) {
    return { status: 409, body: { error: { code: "INVALID_STATUS_TRANSITION", message: err.message } } };
  }
  if (err instanceof Error && err.message.startsWith("Memory not found")) {
    return { status: 404, body: { error: { code: "NOT_FOUND", message: err.message } } };
  }
  return { status: 500, body: { error: { code: "INTERNAL_ERROR", message: (err as Error).message } } };
}

export async function memoryRoutes(app: FastifyInstance, ctx: AppContext) {
  app.get("/api/v1/memories", async (req, reply) => {
    const parsed = listQuerySchema.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400);
      return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
    }
    const q = parsed.data;
    const filter = {
      statuses: q.status ? (q.status.split(",") as any) : undefined,
      kinds: q.kind ? (q.kind.split(",") as any) : undefined,
      scopes: q.scope ? [parseScope(q.scope)] : undefined,
      includeDeleted: q.includeDeleted,
      limit: q.limit,
      cursor: q.cursor,
    };

    if (q.q) {
      const results = await ctx.memoryService.search(q.q, filter, q.limit ?? 50);
      return {
        items: results.map((r) => ({
          memory: serializeMemory(r.memory),
          currentVersion: r.version,
          rawScore: r.rawScore,
        })),
      };
    }

    const page = await ctx.memoryService.list(filter);
    return {
      items: page.items.map((i) => ({
        memory: serializeMemory(i.memory),
        currentVersion: i.currentVersion,
      })),
      nextCursor: page.nextCursor,
    };
  });

  app.post("/api/v1/memories", async (req, reply) => {
    const parsed = createMemorySchema.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400);
      return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
    }
    const input = parsed.data;
    const { memory, version } = await ctx.memoryService.create({
      kind: input.kind,
      scope: input.scope,
      status: input.status,
      confidence: input.confidence,
      importance: input.importance,
      validFrom: input.validFrom,
      validUntil: input.validUntil,
      createdBy: resolveActor(input.actor),
      version: input.version,
    });
    reply.code(201);
    return { memory: serializeMemory(memory), version };
  });

  app.get<{ Params: { id: string } }>("/api/v1/memories/:id", async (req, reply) => {
    const detail = await ctx.memoryService.getDetail(req.params.id);
    if (!detail) {
      reply.code(404);
      return { error: { code: "NOT_FOUND", message: `Memory not found: ${req.params.id}` } };
    }
    return {
      memory: serializeMemory(detail.memory),
      currentVersion: detail.currentVersion,
      versions: detail.versions,
      provenance: detail.provenance,
      reviews: detail.reviews,
      relations: detail.relations,
    };
  });

  app.delete<{ Params: { id: string } }>("/api/v1/memories/:id", async (req, reply) => {
    try {
      const actor = resolveActor(undefined);
      const memory = await ctx.memoryService.delete(req.params.id, actor);
      return { memory: serializeMemory(memory) };
    } catch (err) {
      const { status, body } = errorResponse(err);
      reply.code(status);
      return body;
    }
  });

  app.get<{ Params: { id: string } }>(
    "/api/v1/memories/:id/versions",
    async (req, reply) => {
      const detail = await ctx.memoryService.getDetail(req.params.id);
      if (!detail) {
        reply.code(404);
        return { error: { code: "NOT_FOUND", message: `Memory not found: ${req.params.id}` } };
      }
      return { versions: detail.versions };
    },
  );

  app.post<{ Params: { id: string } }>(
    "/api/v1/memories/:id/versions",
    async (req, reply) => {
      const parsed = addVersionSchema.safeParse(req.body);
      if (!parsed.success) {
        reply.code(400);
        return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
      }
      try {
        const { actor, ...version } = parsed.data;
        const result = await ctx.memoryService.addVersion(req.params.id, {
          ...version,
          createdBy: resolveActor(actor),
        });
        reply.code(201);
        return { memory: serializeMemory(result.memory), version: result.version };
      } catch (err) {
        const { status, body } = errorResponse(err);
        reply.code(status);
        return body;
      }
    },
  );

  app.get<{ Params: { id: string } }>(
    "/api/v1/memories/:id/provenance",
    async (req, reply) => {
      const detail = await ctx.memoryService.getDetail(req.params.id);
      if (!detail) {
        reply.code(404);
        return { error: { code: "NOT_FOUND", message: `Memory not found: ${req.params.id}` } };
      }
      return { provenance: detail.provenance };
    },
  );

  app.post<{ Params: { id: string } }>(
    "/api/v1/memories/:id/provenance",
    async (req, reply) => {
      const parsed = provenanceSchema.safeParse(req.body);
      if (!parsed.success) {
        reply.code(400);
        return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
      }
      const provenance = await ctx.memoryService.addProvenance({
        memoryId: req.params.id,
        ...parsed.data,
      });
      reply.code(201);
      return { provenance };
    },
  );

  app.get<{ Params: { id: string } }>(
    "/api/v1/memories/:id/reviews",
    async (req, reply) => {
      const detail = await ctx.memoryService.getDetail(req.params.id);
      if (!detail) {
        reply.code(404);
        return { error: { code: "NOT_FOUND", message: `Memory not found: ${req.params.id}` } };
      }
      return { reviews: detail.reviews };
    },
  );

  app.post<{ Params: { id: string } }>(
    "/api/v1/memories/:id/reviews",
    async (req, reply) => {
      const parsed = reviewActionSchema.safeParse(req.body);
      if (!parsed.success) {
        reply.code(400);
        return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
      }
      const { action, reason, targetMemoryId, actor: actorInput } = parsed.data;
      const actor = resolveActor(actorInput);
      const memoryId = req.params.id;

      try {
        let memory;
        switch (action) {
          case "approve":
            memory = await ctx.memoryService.approve(memoryId, actor, reason);
            break;
          case "reject":
            memory = await ctx.memoryService.reject(memoryId, actor, reason);
            break;
          case "deprecate":
            memory = await ctx.memoryService.deprecate(memoryId, actor, reason);
            break;
          case "restore":
            memory = await ctx.memoryService.restore(memoryId, actor, reason);
            break;
          case "supersede":
            if (!targetMemoryId) {
              reply.code(400);
              return {
                error: {
                  code: "VALIDATION_ERROR",
                  message: "targetMemoryId (the new memory) is required for supersede",
                },
              };
            }
            memory = await ctx.memoryService.supersede(memoryId, targetMemoryId, actor, reason);
            break;
          case "merge":
            if (!targetMemoryId) {
              reply.code(400);
              return {
                error: {
                  code: "VALIDATION_ERROR",
                  message: "targetMemoryId (the merge destination) is required for merge",
                },
              };
            }
            memory = await ctx.memoryService.merge(memoryId, targetMemoryId, actor, reason);
            break;
        }
        return { memory: serializeMemory(memory!) };
      } catch (err) {
        const { status, body } = errorResponse(err);
        reply.code(status);
        return body;
      }
    },
  );
}

function serializeMemory(memory: {
  id: string;
  kind: string;
  status: string;
  scope: ScopeRef;
  currentVersionId: string;
  confidence?: number;
  importance?: number;
  validFrom?: string;
  validUntil?: string;
  createdAt: string;
  createdBy: unknown;
  updatedAt: string;
  supersededById?: string;
  mergedIntoId?: string;
}) {
  return { ...memory, scopeDisplay: scopeToString(memory.scope) };
}
