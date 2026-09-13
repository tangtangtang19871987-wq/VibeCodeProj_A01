import type {
  CreateMemoryRelationInput,
  CreateProvenanceInput,
  CreateReviewActionInput,
  MemoryRelationRepository,
  ProvenanceRepository,
  ReviewRepository,
} from "@lams/application";
import {
  generateId,
  type ActorRef,
  type MemoryRelation,
  type Provenance,
  type ReviewAction,
} from "@lams/domain";
import type { LamsDatabase } from "../connection.js";
import type { MemoryRelationRow, ProvenanceRow, ReviewActionRow } from "../schema.js";

function actorToColumns(actor: ActorRef) {
  return {
    actor_type: actor.type,
    actor_id: actor.id,
    actor_display_name: actor.displayName ?? null,
  };
}

function createdByColumns(actor: ActorRef) {
  return {
    created_by_type: actor.type,
    created_by_id: actor.id,
    created_by_display_name: actor.displayName ?? null,
  };
}

function rowToProvenance(row: ProvenanceRow): Provenance {
  return {
    id: row.id,
    memoryId: row.memory_id,
    memoryVersionId: row.memory_version_id ?? undefined,
    sourceType: row.source_type as Provenance["sourceType"],
    sourceRef: row.source_ref,
    sessionId: row.session_id ?? undefined,
    eventId: row.event_id ?? undefined,
    excerpt: row.excerpt ?? undefined,
    artifactHash: row.artifact_hash ?? undefined,
    createdAt: row.created_at,
  };
}

export class SqliteProvenanceRepository implements ProvenanceRepository {
  constructor(private readonly db: LamsDatabase) {}

  async create(input: CreateProvenanceInput): Promise<Provenance> {
    const row: ProvenanceRow = {
      id: generateId("provenance"),
      memory_id: input.memoryId,
      memory_version_id: input.memoryVersionId ?? null,
      source_type: input.sourceType,
      source_ref: input.sourceRef,
      session_id: input.sessionId ?? null,
      event_id: input.eventId ?? null,
      excerpt: input.excerpt ?? null,
      artifact_hash: input.artifactHash ?? null,
      created_at: new Date().toISOString(),
    };
    await this.db.insertInto("provenance").values(row).execute();
    return rowToProvenance(row);
  }

  async listByMemoryId(memoryId: string): Promise<Provenance[]> {
    const rows = await this.db
      .selectFrom("provenance")
      .selectAll()
      .where("memory_id", "=", memoryId)
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(rowToProvenance);
  }
}

function rowToReview(row: ReviewActionRow): ReviewAction {
  return {
    id: row.id,
    memoryId: row.memory_id,
    memoryVersionId: row.memory_version_id,
    action: row.action as ReviewAction["action"],
    reason: row.reason ?? undefined,
    actor: {
      type: row.actor_type as ActorRef["type"],
      id: row.actor_id,
      displayName: row.actor_display_name ?? undefined,
    },
    createdAt: row.created_at,
  };
}

export class SqliteReviewRepository implements ReviewRepository {
  constructor(private readonly db: LamsDatabase) {}

  async create(input: CreateReviewActionInput): Promise<ReviewAction> {
    const row: ReviewActionRow = {
      id: generateId("review"),
      memory_id: input.memoryId,
      memory_version_id: input.memoryVersionId,
      action: input.action,
      reason: input.reason ?? null,
      ...actorToColumns(input.actor),
      created_at: new Date().toISOString(),
    };
    await this.db.insertInto("review_actions").values(row).execute();
    return rowToReview(row);
  }

  async listByMemoryId(memoryId: string): Promise<ReviewAction[]> {
    const rows = await this.db
      .selectFrom("review_actions")
      .selectAll()
      .where("memory_id", "=", memoryId)
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(rowToReview);
  }
}

function rowToRelation(row: MemoryRelationRow): MemoryRelation {
  return {
    id: row.id,
    fromMemoryId: row.from_memory_id,
    toMemoryId: row.to_memory_id,
    type: row.type as MemoryRelation["type"],
    note: row.note ?? undefined,
    createdAt: row.created_at,
    createdBy: {
      type: row.created_by_type as ActorRef["type"],
      id: row.created_by_id,
      displayName: row.created_by_display_name ?? undefined,
    },
  };
}

export class SqliteMemoryRelationRepository implements MemoryRelationRepository {
  constructor(private readonly db: LamsDatabase) {}

  async create(input: CreateMemoryRelationInput): Promise<MemoryRelation> {
    const row: MemoryRelationRow = {
      id: generateId("relation"),
      from_memory_id: input.fromMemoryId,
      to_memory_id: input.toMemoryId,
      type: input.type,
      note: input.note ?? null,
      ...createdByColumns(input.createdBy),
      created_at: new Date().toISOString(),
    };
    await this.db.insertInto("memory_relations").values(row).execute();
    return rowToRelation(row);
  }

  async listForMemory(memoryId: string): Promise<MemoryRelation[]> {
    const rows = await this.db
      .selectFrom("memory_relations")
      .selectAll()
      .where((eb) =>
        eb.or([eb("from_memory_id", "=", memoryId), eb("to_memory_id", "=", memoryId)]),
      )
      .orderBy("created_at", "asc")
      .execute();
    return rows.map(rowToRelation);
  }
}
