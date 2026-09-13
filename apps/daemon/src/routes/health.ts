import type { FastifyInstance } from "fastify";
import type { AppContext } from "../context.js";

export async function healthRoutes(app: FastifyInstance, ctx: AppContext) {
  app.get("/api/v1/health", async (_req, reply) => {
    const report = await ctx.healthService.check();
    reply.code(report.ok ? 200 : 503);
    return report;
  });
}
