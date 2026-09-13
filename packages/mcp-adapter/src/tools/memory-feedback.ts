import { FEEDBACK_EVENT_TYPES } from "@lams/domain";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import type { McpToolContext } from "../context.js";
import { textResult } from "../result.js";

const MAX_MEMORY_IDS = 10;

const inputShape = {
  session_id: z.string(),
  trace_id: z.string().optional().describe("The recall trace this feedback relates to, if any."),
  memory_ids: z.array(z.string()).min(1).max(MAX_MEMORY_IDS),
  event_type: z.enum(FEEDBACK_EVENT_TYPES),
  note: z.string().max(1000).optional(),
  outcome_metric: z.record(z.number()).optional(),
};

export function registerMemoryFeedback(server: McpServer, ctx: McpToolContext) {
  server.registerTool(
    "memory_feedback",
    {
      title: "Report delivery, usefulness, or outcome feedback for memories",
      description:
        "Creates persistent data: every call is recorded as 'reported' evidence (Section 9.7) — never claimed as directly observed by the daemon, since it always comes from an explicit report. Use this to close the loop after memory_recall.",
      inputSchema: inputShape,
    },
    async (args) => {
      const results = [];
      for (const memoryId of args.memory_ids) {
        const record = await ctx.feedbackService.record({
          sessionId: args.session_id,
          recallTraceId: args.trace_id,
          memoryId,
          eventType: args.event_type,
          note: args.note,
          outcomeMetric: args.outcome_metric,
        });
        results.push({ memory_id: memoryId, feedback_id: record.id, evidenceLevel: record.evidenceLevel });
      }
      return textResult({ recorded: results });
    },
  );
}
