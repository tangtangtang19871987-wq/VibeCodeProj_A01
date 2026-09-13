import type {
  AddVersionInput,
  CreateMemoryInput,
  MemoryListFilter,
  MemoryListPage,
  MemoryRepository,
} from "@lams/application";
import {
  generateId,
  type ActorRef,
  type Memory,
  type MemoryStatus,
  type MemoryVersion,
  type ScopeRef,
} from "@lams/domain";
import type { LamsDatabase } from "../connection.js";
import type { MemoryRow, MemoryVersionRow } from "../schema.js";

function actorToColumns(actor: ActorRef) {
  return {
    created_by_type: actor.type,
    created_by_id: actor.id,
    created_by_display_name: actor.displayName ?? null,
  };
}

function columnsToActor(row: {
  created_by_type: string;
  created_by_id: string;
  created_by_display_name: string | null;
}): ActorRef {
  return {
    type: row.created_by_type as ActorRef["type"],
    id: row.created_by_id,
    displayName: row.created_by_display_name ?? undefined,
  };
}

function rowToMemory(row: MemoryRow): Memory {
  const scope: ScopeRef = { level: row.scope_level as ScopeRef["level"], id: row.scope_id ?? undefined };
  return {
    id: row.id,
    kind: row.kind as Memory["kind"],
    status: row.status as MemoryStatus,
    scope,
    currentVersionId: row.current_version_id,
    confidence: row.confidence ?? undefined,
    importance: row.importance ?? undefined,
    validFrom: row.valid_from ?? undefined,
    validUntil: row.valid_until ?? undefined,
    createdAt: row.created_at,
    createdBy: columnsToActor(row),
    updatedAt: row.updated_at,
    supersededById: row.superseded_by_id ?? undefined,
    mergedIntoId: row.merged_into_id ?? undefined,
  };
}

function rowToVersion(row: MemoryVersionRow): MemoryVersion {
  return {
    id: row.id,
    memoryId: row.memory_id,
    version: row.version,
    title: row.title,
    content: row.content,
    summary: row.summary ?? undefined,
    tags: JSON.parse(row.tags) as string[],
    structuredData: row.structured_data
      ? (JSON.parse(row.structured_data) as Record<string, unknown>)
      : undefined,
    changeReason: row.change_reason ?? undefined,
    createdAt: row.created_at,
    createdBy: columnsToActor(row),
  };
}

export class SqliteMemoryRepository implements MemoryRepository {
  constructor(private readonly db: LamsDatabase) {}

  async createWithFirstVersion(
    input: CreateMemoryInput,
  ): Promise<{ memory: Memory; version: MemoryVersion }> {
    const now = new Date().toISOString();
    const memoryId = generateId("memory");
    const versionId = generateId("memoryVersion");

    const versionRow: MemoryVersionRow = {
      id: versionId,
      memory_id: memoryId,
      version: 1,
      title: input.version.title,
      content: input.version.content,
      summary: input.version.summary ?? null,
      tags: JSON.stringify(input.version.tags ?? []),
      structured_data: input.version.structuredData
        ? JSON.stringify(input.version.structuredData)
        : null,
      change_reason: input.version.changeReason ?? null,
      created_at: now,
      ...actorToColumns(input.createdBy),
    };

    const memoryRow: MemoryRow = {
      id: memoryId,
      kind: input.kind,
      status: input.status ?? "draft",
      scope_level: input.scope.level,
      scope_id: input.scope.id ?? null,
      current_version_id: versionId,
      confidence: input.confidence ?? null,
      importance: input.importance ?? null,
      valid_from: input.validFrom ?? null,
      valid_until: input.validUntil ?? null,
      created_at: now,
      ...actorToColumns(input.createdBy),
      updated_at: now,
      superseded_by_id: null,
      merged_into_id: null,
    };

    await this.db.transaction().execute(async (trx) => {
      // memories.current_version_id is not FK-constrained (no circular FK
      // in SQLite without deferred constraints), so the memory row can be
      // inserted first; memory_versions.memory_id *is* FK-constrained, so
      // it must come second.
      await trx.insertInto("memories").values(memoryRow).execute();
      await trx.insertInto("memory_versions").values(versionRow).execute();
    });

    return { memory: rowToMemory(memoryRow), version: rowToVersion(versionRow) };
  }

  async getById(id: string): Promise<Memory | null> {
    const row = await this.db
      .selectFrom("memories")
      .selectAll()
      .where("id", "=", id)
      .executeTakeFirst();
    return row ? rowToMemory(row) : null;
  }

  async getCurrentVersion(memoryId: string): Promise<MemoryVersion | null> {
    const memory = await this.db
      .selectFrom("memories")
      .select("current_version_id")
      .where("id", "=", memoryId)
      .executeTakeFirst();
    if (!memory) return null;
    return this.getVersion(memory.current_version_id);
  }

  async getVersion(versionId: string): Promise<MemoryVersion | null> {
    const row = await this.db
      .selectFrom("memory_versions")
      .selectAll()
      .where("id", "=", versionId)
      .executeTakeFirst();
    return row ? rowToVersion(row) : null;
  }

  async listVersions(memoryId: string): Promise<MemoryVersion[]> {
    const rows = await this.db
      .selectFrom("memory_versions")
      .selectAll()
      .where("memory_id", "=", memoryId)
      .orderBy("version", "asc")
      .execute();
    return rows.map(rowToVersion);
  }

  async addVersion(
    memoryId: string,
    input: AddVersionInput,
  ): Promise<{ memory: Memory; version: MemoryVersion }> {
    const now = new Date().toISOString();

    return this.db.transaction().execute(async (trx) => {
      const existing = await trx
        .selectFrom("memories")
        .selectAll()
        .where("id", "=", memoryId)
        .executeTakeFirstOrThrow();

      const latest = await trx
        .selectFrom("memory_versions")
        .select("version")
        .where("memory_id", "=", memoryId)
        .orderBy("version", "desc")
        .executeTakeFirstOrThrow();

      const versionRow: MemoryVersionRow = {
        id: generateId("memoryVersion"),
        memory_id: memoryId,
        version: latest.version + 1,
        title: input.title,
        content: input.content,
        summary: input.summary ?? null,
        tags: JSON.stringify(input.tags ?? []),
        structured_data: input.structuredData
          ? JSON.stringify(input.structuredData)
          : null,
        change_reason: input.changeReason ?? null,
        created_at: now,
        ...actorToColumns(input.createdBy),
      };
      await trx.insertInto("memory_versions").values(versionRow).execute();

      await trx
        .updateTable("memories")
        .set({ current_version_id: versionRow.id, updated_at: now })
        .where("id", "=", memoryId)
        .execute();

      return {
        memory: rowToMemory({ ...existing, current_version_id: versionRow.id, updated_at: now }),
        version: rowToVersion(versionRow),
      };
    });
  }

  async updateStatus(
    memoryId: string,
    status: MemoryStatus,
    extra?: { supersededById?: string; mergedIntoId?: string },
  ): Promise<Memory> {
    const now = new Date().toISOString();
    await this.db
      .updateTable("memories")
      .set({
        status,
        updated_at: now,
        ...(extra?.supersededById ? { superseded_by_id: extra.supersededById } : {}),
        ...(extra?.mergedIntoId ? { merged_into_id: extra.mergedIntoId } : {}),
      })
      .where("id", "=", memoryId)
      .execute();
    const row = await this.db
      .selectFrom("memories")
      .selectAll()
      .where("id", "=", memoryId)
      .executeTakeFirstOrThrow();
    return rowToMemory(row);
  }

  async list(filter: MemoryListFilter): Promise<MemoryListPage> {
    let q = this.db.selectFrom("memories").selectAll();

    if (!filter.includeDeleted) {
      q = q.where("status", "!=", "deleted");
    }
    if (filter.statuses && filter.statuses.length > 0) {
      q = q.where("status", "in", filter.statuses);
    }
    if (filter.kinds && filter.kinds.length > 0) {
      q = q.where("kind", "in", filter.kinds);
    }
    if (filter.scopes && filter.scopes.length > 0) {
      q = q.where((eb) =>
        eb.or(
          filter.scopes!.map((s) =>
            s.id
              ? eb.and([eb("scope_level", "=", s.level), eb("scope_id", "=", s.id)])
              : eb("scope_level", "=", s.level),
          ),
        ),
      );
    }
    if (filter.cursor) {
      q = q.where("id", "<", filter.cursor);
    }

    const limit = filter.limit ?? 50;
    const rows = await q.orderBy("id", "desc").limit(limit + 1).execute();
    const page = rows.slice(0, limit);
    const nextCursor = rows.length > limit ? page[page.length - 1]?.id : undefined;

    const items = await Promise.all(
      page.map(async (row) => {
        const memory = rowToMemory(row);
        const version = await this.getVersion(row.current_version_id);
        return { memory, currentVersion: version! };
      }),
    );

    return { items, nextCursor };
  }
}
