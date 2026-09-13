import type { RecallRepository, RecordDeliveryEventInput, DeliveryRepository } from "@lams/application";
import {
  generateId,
  type DeliveryEvent,
  type DeliveryStage,
  type EvidenceLevel,
  type MemoryKind,
  type MemoryStatus,
  type RecallCandidate,
  type RecallReasonCode,
  type RecallTrace,
  type ScopeFilter,
} from "@lams/domain";
import type { LamsDatabase } from "../connection.js";
import type { DeliveryEventRow, RecallCandidateRow, RecallTraceRow } from "../schema.js";

function traceToRow(trace: RecallTrace): RecallTraceRow {
  return {
    id: trace.id,
    session_id: trace.sessionId,
    query: trace.query,
    intent: trace.intent ?? null,
    scope_filter: JSON.stringify(trace.scopeFilter),
    kind_filter: trace.kindFilter ? JSON.stringify(trace.kindFilter) : null,
    status_filter: JSON.stringify(trace.statusFilter),
    strategy: trace.strategy,
    strategy_version: trace.strategyVersion,
    max_items: trace.maxItems,
    budget_max_items: trace.budget.maxItems,
    budget_max_chars: trace.budget.maxChars,
    candidate_count: trace.candidateCount,
    selected_count: trace.selectedCount,
    returned_count: trace.returnedCount,
    returned_chars: trace.returnedChars,
    estimated_tokens: trace.estimatedTokens ?? null,
    latency_ms: trace.latencyMs,
    created_at: trace.createdAt,
  };
}

function rowToTrace(row: RecallTraceRow): RecallTrace {
  return {
    id: row.id,
    sessionId: row.session_id,
    query: row.query,
    intent: row.intent ?? undefined,
    scopeFilter: JSON.parse(row.scope_filter) as ScopeFilter,
    kindFilter: row.kind_filter ? (JSON.parse(row.kind_filter) as MemoryKind[]) : undefined,
    statusFilter: JSON.parse(row.status_filter) as MemoryStatus[],
    strategy: row.strategy,
    strategyVersion: row.strategy_version,
    maxItems: row.max_items,
    budget: { maxItems: row.budget_max_items, maxChars: row.budget_max_chars },
    candidateCount: row.candidate_count,
    selectedCount: row.selected_count,
    returnedCount: row.returned_count,
    returnedChars: row.returned_chars,
    estimatedTokens: row.estimated_tokens ?? undefined,
    latencyMs: row.latency_ms,
    createdAt: row.created_at,
  };
}

function candidateToRow(c: RecallCandidate): RecallCandidateRow {
  return {
    id: c.id,
    recall_trace_id: c.recallTraceId,
    memory_id: c.memoryId,
    memory_version_id: c.memoryVersionId,
    eligible: c.eligible ? 1 : 0,
    raw_score: c.rawScore,
    score_components: JSON.stringify(c.scoreComponents),
    final_score: c.finalScore,
    rank: c.rank,
    selected: c.selected ? 1 : 0,
    returned: c.returned ? 1 : 0,
    exclusion_reasons: JSON.stringify(c.exclusionReasons),
    status_at_query_time: c.statusAtQueryTime,
    reason_codes: JSON.stringify(c.reasonCodes),
  };
}

function rowToCandidate(row: RecallCandidateRow): RecallCandidate {
  return {
    id: row.id,
    recallTraceId: row.recall_trace_id,
    memoryId: row.memory_id,
    memoryVersionId: row.memory_version_id,
    eligible: !!row.eligible,
    rawScore: row.raw_score,
    scoreComponents: JSON.parse(row.score_components),
    finalScore: row.final_score,
    rank: row.rank,
    selected: !!row.selected,
    returned: !!row.returned,
    exclusionReasons: JSON.parse(row.exclusion_reasons) as RecallReasonCode[],
    statusAtQueryTime: row.status_at_query_time as MemoryStatus,
    reasonCodes: JSON.parse(row.reason_codes) as RecallReasonCode[],
  };
}

export class SqliteRecallRepository implements RecallRepository {
  constructor(private readonly db: LamsDatabase) {}

  async createTraceWithCandidates(
    trace: RecallTrace,
    candidates: RecallCandidate[],
  ): Promise<void> {
    await this.db.transaction().execute(async (trx) => {
      await trx.insertInto("recall_traces").values(traceToRow(trace)).execute();
      if (candidates.length > 0) {
        await trx
          .insertInto("recall_candidates")
          .values(candidates.map(candidateToRow))
          .execute();
      }
    });
  }

  async getTrace(id: string): Promise<RecallTrace | null> {
    const row = await this.db
      .selectFrom("recall_traces")
      .selectAll()
      .where("id", "=", id)
      .executeTakeFirst();
    return row ? rowToTrace(row) : null;
  }

  async listCandidates(traceId: string): Promise<RecallCandidate[]> {
    const rows = await this.db
      .selectFrom("recall_candidates")
      .selectAll()
      .where("recall_trace_id", "=", traceId)
      .orderBy("rank", "asc")
      .execute();
    return rows.map(rowToCandidate);
  }

  async listTracesForSession(sessionId: string): Promise<RecallTrace[]> {
    const rows = await this.db
      .selectFrom("recall_traces")
      .selectAll()
      .where("session_id", "=", sessionId)
      .orderBy("created_at", "desc")
      .execute();
    return rows.map(rowToTrace);
  }
}

function rowToDelivery(row: DeliveryEventRow): DeliveryEvent {
  return {
    id: row.id,
    sessionId: row.session_id,
    recallTraceId: row.recall_trace_id ?? undefined,
    memoryId: row.memory_id,
    memoryVersionId: row.memory_version_id ?? undefined,
    stage: row.stage as DeliveryStage,
    evidenceLevel: row.evidence_level as EvidenceLevel,
    note: row.note ?? undefined,
    createdAt: row.created_at,
  };
}

export class SqliteDeliveryRepository implements DeliveryRepository {
  constructor(private readonly db: LamsDatabase) {}

  async record(input: RecordDeliveryEventInput): Promise<DeliveryEvent> {
    const row: DeliveryEventRow = {
      id: generateId("deliveryEvent"),
      session_id: input.sessionId,
      recall_trace_id: input.recallTraceId ?? null,
      memory_id: input.memoryId,
      memory_version_id: input.memoryVersionId ?? null,
      stage: input.stage,
      evidence_level: input.evidenceLevel,
      note: input.note ?? null,
      created_at: new Date().toISOString(),
    };
    await this.db.insertInto("delivery_events").values(row).execute();
    return rowToDelivery(row);
  }

  async listForSession(sessionId: string): Promise<DeliveryEvent[]> {
    const rows = await this.db
      .selectFrom("delivery_events")
      .selectAll()
      .where("session_id", "=", sessionId)
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(rowToDelivery);
  }

  async listForTrace(recallTraceId: string): Promise<DeliveryEvent[]> {
    const rows = await this.db
      .selectFrom("delivery_events")
      .selectAll()
      .where("recall_trace_id", "=", recallTraceId)
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(rowToDelivery);
  }
}
