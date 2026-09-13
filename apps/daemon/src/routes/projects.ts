import type { FastifyInstance } from "fastify";
import { z } from "zod";
import type { AppContext } from "../context.js";

const createProjectSchema = z.object({
  key: z.string().min(1).max(200),
  name: z.string().min(1).max(200),
  description: z.string().max(2000).optional(),
});

export async function projectRoutes(app: FastifyInstance, ctx: AppContext) {
  app.get("/api/v1/projects", async () => {
    return { projects: await ctx.projectService.list() };
  });

  app.post("/api/v1/projects", async (req, reply) => {
    const parsed = createProjectSchema.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400);
      return {
        error: { code: "VALIDATION_ERROR", message: parsed.error.message },
      };
    }
    const project = await ctx.projectService.create(parsed.data);
    reply.code(201);
    return { project };
  });
}
