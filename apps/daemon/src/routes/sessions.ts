import { SESSION_STATUSES } from "@lams/domain";
import type { FastifyInstance } from "fastify";
import { z } from "zod";
import type { AppContext } from "../context.js";

const openSessionSchema = z.object({
  harness: z.string().min(1).max(100),
  projectId: z.string().min(1),
  agentName: z.string().max(200).optional(),
  externalSessionId: z.string().max(200).optional(),
  title: z.string().max(300).optional(),
  taskSummary: z.string().max(2000).optional(),
  metadata: z.record(z.unknown()).optional(),
});

const updateSessionSchema = z.object({
  title: z.string().max(300).optional(),
  taskSummary: z.string().max(2000).optional(),
  metadata: z.record(z.unknown()).optional(),
});

const closeSessionSchema = z.object({
  status: z.enum(["completed", "failed", "abandoned"]),
  outcome: z
    .object({
      result: z.enum(["success", "failure", "partial", "unknown"]),
      summary: z.string().max(2000).optional(),
      metric: z.record(z.number()).optional(),
    })
    .optional(),
});

const listQuerySchema = z.object({
  projectId: z.string().optional(),
  status: z.string().optional(),
});

export async function sessionRoutes(app: FastifyInstance, ctx: AppContext) {
  app.post("/api/v1/sessions", async (req, reply) => {
    const parsed = openSessionSchema.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400);
      return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
    }
    const session = await ctx.sessionService.open(parsed.data);
    reply.code(201);
    return { session };
  });

  app.get("/api/v1/sessions", async (req, reply) => {
    const parsed = listQuerySchema.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400);
      return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
    }
    const status = parsed.data.status
      ? (parsed.data.status.split(",") as (typeof SESSION_STATUSES)[number][])
      : undefined;
    const sessions = await ctx.sessionService.list({ projectId: parsed.data.projectId, status });
    return { sessions };
  });

  app.get<{ Params: { id: string } }>("/api/v1/sessions/:id", async (req, reply) => {
    const session = await ctx.sessionService.getById(req.params.id);
    if (!session) {
      reply.code(404);
      return { error: { code: "NOT_FOUND", message: `Session not found: ${req.params.id}` } };
    }
    const events = await ctx.sessionService.listEvents(req.params.id);
    return { session, events };
  });

  app.patch<{ Params: { id: string } }>("/api/v1/sessions/:id", async (req, reply) => {
    const parsed = updateSessionSchema.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400);
      return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
    }
    const session = await ctx.sessionService.update(req.params.id, parsed.data);
    return { session };
  });

  app.post<{ Params: { id: string } }>("/api/v1/sessions/:id/close", async (req, reply) => {
    const parsed = closeSessionSchema.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400);
      return { error: { code: "VALIDATION_ERROR", message: parsed.error.message } };
    }
    const session = await ctx.sessionService.close(
      req.params.id,
      parsed.data.status,
      parsed.data.outcome,
    );
    return { session };
  });
}
