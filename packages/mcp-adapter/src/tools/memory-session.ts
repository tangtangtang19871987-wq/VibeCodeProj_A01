import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { McpToolContext } from "../context.js";
import { errorResult, textResult } from "../result.js";

const inputShape = {
  action: z.enum(["open", "update", "close"]).describe("Which session lifecycle operation to perform."),
  session_id: z.string().optional().describe("Required for update/close; ignored for open."),
  external_session_id: z.string().optional().describe("The harness's own session identifier, if it has one."),
  project: z.string().optional().describe("Project key, e.g. \"ilt-agent\". Required for action=open; the project is created if it doesn't exist yet."),
  harness: z.string().optional().describe("Harness name, e.g. \"opencode\". Required for action=open."),
  agent_name: z.string().optional(),
  title: z.string().optional(),
  task_summary: z.string().optional(),
  status: z.enum(["completed", "failed", "abandoned"]).optional().describe("Required for action=close."),
  outcome: z
    .object({
      result: z.enum(["success", "failure", "partial", "unknown"]),
      summary: z.string().optional(),
      metric: z.record(z.number()).optional(),
    })
    .optional(),
};

export function registerMemorySession(server: McpServer, ctx: McpToolContext) {
  server.registerTool(
    "memory_session",
    {
      title: "Open, update, or close a memory-tracked agent session",
      description:
        "Creates persistent data: opening or updating a session records a session_opened/session_updated event; closing one records the outcome. Every memory_recall/memory_remember/memory_feedback call must reference a session_id from here.",
      inputSchema: inputShape,
    },
    async (args) => {
      try {
        if (args.action === "open") {
          if (!args.project || !args.harness) {
            return errorResult("VALIDATION_ERROR", "project and harness are required to open a session.");
          }
          const project = await ctx.projectService.create({ key: args.project, name: args.project });
          const session = await ctx.sessionService.open({
            harness: args.harness,
            projectId: project.id,
            agentName: args.agent_name,
            externalSessionId: args.external_session_id,
            title: args.title,
            taskSummary: args.task_summary,
          });
          return textResult({ session_id: session.id, status: session.status, project_id: project.id });
        }

        if (!args.session_id) {
          return errorResult("VALIDATION_ERROR", "session_id is required for update/close.");
        }

        if (args.action === "update") {
          const session = await ctx.sessionService.update(args.session_id, {
            title: args.title,
            taskSummary: args.task_summary,
          });
          return textResult({ session_id: session.id, status: session.status });
        }

        // action === "close"
        if (!args.status) {
          return errorResult("VALIDATION_ERROR", "status is required for action=close.");
        }
        const session = await ctx.sessionService.close(args.session_id, args.status, args.outcome);
        return textResult({ session_id: session.id, status: session.status });
      } catch (err) {
        return errorResult("NOT_FOUND", (err as Error).message);
      }
    },
  );
}
