import type {
  OpenSessionInput,
  SessionListFilter,
  SessionRepository,
  UpdateSessionInput,
} from "@lams/application";
import {
  generateId,
  type AgentSession,
  type SessionEvent,
  type SessionEventType,
  type SessionOutcome,
  type SessionStatus,
} from "@lams/domain";
import type { LamsDatabase } from "../connection.js";
import type { SessionEventRow, SessionRow } from "../schema.js";

function rowToSession(row: SessionRow): AgentSession {
  return {
    id: row.id,
    externalSessionId: row.external_session_id ?? undefined,
    harness: row.harness,
    agentName: row.agent_name ?? undefined,
    projectId: row.project_id,
    title: row.title ?? undefined,
    taskSummary: row.task_summary ?? undefined,
    status: row.status as SessionStatus,
    startedAt: row.started_at,
    endedAt: row.ended_at ?? undefined,
    outcome: row.outcome_result
      ? {
          result: row.outcome_result as SessionOutcome["result"],
          summary: row.outcome_summary ?? undefined,
          metric: row.outcome_metric ? JSON.parse(row.outcome_metric) : undefined,
        }
      : undefined,
    metadata: row.metadata ? JSON.parse(row.metadata) : undefined,
  };
}

function rowToEvent(row: SessionEventRow): SessionEvent {
  return {
    id: row.id,
    sessionId: row.session_id,
    type: row.type as SessionEventType,
    payloadSchemaVersion: row.payload_schema_version,
    payload: JSON.parse(row.payload),
    createdAt: row.created_at,
  };
}

export class SqliteSessionRepository implements SessionRepository {
  constructor(private readonly db: LamsDatabase) {}

  async open(input: OpenSessionInput): Promise<AgentSession> {
    const row: SessionRow = {
      id: generateId("session"),
      external_session_id: input.externalSessionId ?? null,
      harness: input.harness,
      agent_name: input.agentName ?? null,
      project_id: input.projectId,
      title: input.title ?? null,
      task_summary: input.taskSummary ?? null,
      status: "open",
      started_at: new Date().toISOString(),
      ended_at: null,
      outcome_result: null,
      outcome_summary: null,
      outcome_metric: null,
      metadata: input.metadata ? JSON.stringify(input.metadata) : null,
    };
    await this.db.insertInto("sessions").values(row).execute();
    return rowToSession(row);
  }

  async getById(id: string): Promise<AgentSession | null> {
    const row = await this.db
      .selectFrom("sessions")
      .selectAll()
      .where("id", "=", id)
      .executeTakeFirst();
    return row ? rowToSession(row) : null;
  }

  async update(id: string, patch: UpdateSessionInput): Promise<AgentSession> {
    await this.db
      .updateTable("sessions")
      .set({
        ...(patch.title !== undefined ? { title: patch.title } : {}),
        ...(patch.taskSummary !== undefined ? { task_summary: patch.taskSummary } : {}),
        ...(patch.metadata !== undefined ? { metadata: JSON.stringify(patch.metadata) } : {}),
      })
      .where("id", "=", id)
      .execute();
    const row = await this.db
      .selectFrom("sessions")
      .selectAll()
      .where("id", "=", id)
      .executeTakeFirstOrThrow();
    return rowToSession(row);
  }

  async close(
    id: string,
    status: SessionStatus,
    outcome?: SessionOutcome,
  ): Promise<AgentSession> {
    await this.db
      .updateTable("sessions")
      .set({
        status,
        ended_at: new Date().toISOString(),
        outcome_result: outcome?.result ?? null,
        outcome_summary: outcome?.summary ?? null,
        outcome_metric: outcome?.metric ? JSON.stringify(outcome.metric) : null,
      })
      .where("id", "=", id)
      .execute();
    const row = await this.db
      .selectFrom("sessions")
      .selectAll()
      .where("id", "=", id)
      .executeTakeFirstOrThrow();
    return rowToSession(row);
  }

  async list(filter: SessionListFilter): Promise<AgentSession[]> {
    let q = this.db.selectFrom("sessions").selectAll();
    if (filter.projectId) q = q.where("project_id", "=", filter.projectId);
    if (filter.status && filter.status.length > 0) q = q.where("status", "in", filter.status);
    const rows = await q
      .orderBy("started_at", "desc")
      .limit(filter.limit ?? 50)
      .execute();
    return rows.map(rowToSession);
  }

  async appendEvent(
    sessionId: string,
    type: SessionEventType,
    payload: Record<string, unknown>,
  ): Promise<SessionEvent> {
    const row: SessionEventRow = {
      id: generateId("event"),
      session_id: sessionId,
      type,
      payload_schema_version: 1,
      payload: JSON.stringify(payload),
      created_at: new Date().toISOString(),
    };
    await this.db.insertInto("session_events").values(row).execute();
    return rowToEvent(row);
  }

  async listEvents(sessionId: string): Promise<SessionEvent[]> {
    const rows = await this.db
      .selectFrom("session_events")
      .selectAll()
      .where("session_id", "=", sessionId)
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(rowToEvent);
  }
}
