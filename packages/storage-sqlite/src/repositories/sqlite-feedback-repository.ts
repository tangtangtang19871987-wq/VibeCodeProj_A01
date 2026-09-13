import type { CreateFeedbackInput, FeedbackRepository } from "@lams/application";
import {
  generateId,
  type EvidenceLevel,
  type FeedbackEventType,
  type MemoryFeedback,
} from "@lams/domain";
import type { LamsDatabase } from "../connection.js";
import type { MemoryFeedbackRow } from "../schema.js";

function rowToFeedback(row: MemoryFeedbackRow): MemoryFeedback {
  return {
    id: row.id,
    sessionId: row.session_id,
    recallTraceId: row.recall_trace_id ?? undefined,
    memoryId: row.memory_id,
    eventType: row.event_type as FeedbackEventType,
    evidenceLevel: row.evidence_level as EvidenceLevel,
    note: row.note ?? undefined,
    outcomeMetric: row.outcome_metric ? JSON.parse(row.outcome_metric) : undefined,
    createdAt: row.created_at,
  };
}

export class SqliteFeedbackRepository implements FeedbackRepository {
  constructor(private readonly db: LamsDatabase) {}

  async create(input: CreateFeedbackInput): Promise<MemoryFeedback> {
    const row: MemoryFeedbackRow = {
      id: generateId("feedback"),
      session_id: input.sessionId,
      recall_trace_id: input.recallTraceId ?? null,
      memory_id: input.memoryId,
      event_type: input.eventType,
      evidence_level: input.evidenceLevel,
      note: input.note ?? null,
      outcome_metric: input.outcomeMetric ? JSON.stringify(input.outcomeMetric) : null,
      created_at: new Date().toISOString(),
    };
    await this.db.insertInto("memory_feedback").values(row).execute();
    return rowToFeedback(row);
  }

  async listForMemory(memoryId: string): Promise<MemoryFeedback[]> {
    const rows = await this.db
      .selectFrom("memory_feedback")
      .selectAll()
      .where("memory_id", "=", memoryId)
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(rowToFeedback);
  }

  async listForSession(sessionId: string): Promise<MemoryFeedback[]> {
    const rows = await this.db
      .selectFrom("memory_feedback")
      .selectAll()
      .where("session_id", "=", sessionId)
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(rowToFeedback);
  }
}
