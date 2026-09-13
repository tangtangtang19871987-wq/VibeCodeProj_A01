import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyInstance } from "fastify";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { AppContext } from "./context.js";
import { healthRoutes } from "./routes/health.js";
import { memoryRoutes } from "./routes/memories.js";
import { projectRoutes } from "./routes/projects.js";

const currentDir = path.dirname(fileURLToPath(import.meta.url));
// apps/daemon/{src,dist}/.. -> apps/daemon -> apps -> <repo root>
const STUDIO_DIST = path.resolve(currentDir, "../../studio/dist");

export async function buildApp(ctx: AppContext): Promise<FastifyInstance> {
  const app = Fastify({
    logger: {
      level: process.env.LAMS_LOG_LEVEL ?? "info",
      transport:
        process.env.NODE_ENV === "production"
          ? undefined
          : { target: "pino-pretty", options: { singleLine: true } },
    },
  });

  await healthRoutes(app, ctx);
  await projectRoutes(app, ctx);
  await memoryRoutes(app, ctx);

  // ADR 0007: in production, serve the prebuilt studio UI from the same
  // Fastify instance/port as the API. In development the UI runs under its
  // own Vite dev server and proxies /api/* here, so this block is a no-op.
  if (fs.existsSync(STUDIO_DIST)) {
    await app.register(fastifyStatic, {
      root: STUDIO_DIST,
      wildcard: false,
    });
    app.setNotFoundHandler(async (req, reply) => {
      if (req.raw.url?.startsWith("/api/") || req.raw.url?.startsWith("/mcp")) {
        reply.code(404);
        return { error: { code: "NOT_FOUND", message: "Route not found" } };
      }
      return reply.sendFile("index.html");
    });
  }

  return app;
}
