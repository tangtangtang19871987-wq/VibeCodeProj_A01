import {
  assertValidTransition,
  RECALL_ELIGIBLE_STATUSES,
  type ActorRef,
  type Memory,
  type MemoryStatus,
  type MemoryVersion,
  type ScopeRef,
} from "@lams/domain";
import type {
  AddVersionInput,
  CreateMemoryInput,
  FtsSearchPort,
  MemoryListFilter,
  MemoryListPage,
  MemoryRepository,
} from "../ports/memory.js";
import type {
  CreateMemoryRelationInput,
  MemoryRelationRepository,
  ProvenanceRepository,
  ReviewRepository,
} from "../ports/provenance-review.js";

export type MemoryDetail = {
  memory: Memory;
  currentVersion: MemoryVersion;
  versions: MemoryVersion[];
  provenance: Awaited<ReturnType<ProvenanceRepository["listByMemoryId"]>>;
  reviews: Awaited<ReturnType<ReviewRepository["listByMemoryId"]>>;
  relations: Awaited<ReturnType<MemoryRelationRepository["listForMemory"]>>;
};

export type SearchResultItem = {
  memory: Memory;
  version: MemoryVersion;
  rawScore: number;
};

/**
 * Orchestrates the memory lifecycle (Section 9, 15). This is the only place
 * allowed to mutate memory status — every transition goes through
 * `assertValidTransition` and, where the PRD defines a reviewable action
 * (Section 9.4), records a `ReviewAction` so the audit trail is complete by
 * construction rather than by callers remembering to log it.
 */
export class MemoryService {
  constructor(
    private readonly memories: MemoryRepository,
    private readonly provenanceRepo: ProvenanceRepository,
    private readonly reviews: ReviewRepository,
    private readonly relations: MemoryRelationRepository,
    private readonly ftsSearch: FtsSearchPort,
  ) {}

  async create(
    input: CreateMemoryInput,
  ): Promise<{ memory: Memory; version: MemoryVersion }> {
    // Section 15.1: agent-created memories are always draft by default,
    // regardless of what the caller requests. Human/import callers may
    // request "verified" explicitly.
    const status: MemoryStatus =
      input.createdBy.type === "agent" ? "draft" : (input.status ?? "draft");
    return this.memories.createWithFirstVersion({ ...input, status });
  }

  async addProvenance(input: Parameters<ProvenanceRepository["create"]>[0]) {
    return this.provenanceRepo.create(input);
  }

  async addVersion(
    memoryId: string,
    input: AddVersionInput,
  ): Promise<{ memory: Memory; version: MemoryVersion }> {
    const result = await this.memories.addVersion(memoryId, input);
    await this.reviews.create({
      memoryId,
      memoryVersionId: result.version.id,
      action: "edit",
      reason: input.changeReason,
      actor: input.createdBy,
    });
    return result;
  }

  async getDetail(memoryId: string): Promise<MemoryDetail | null> {
    const memory = await this.memories.getById(memoryId);
    if (!memory) return null;
    const [currentVersion, versions, provenance, reviews, relations] =
      await Promise.all([
        this.memories.getCurrentVersion(memoryId),
        this.memories.listVersions(memoryId),
        this.provenanceRepo.listByMemoryId(memoryId),
        this.reviews.listByMemoryId(memoryId),
        this.relations.listForMemory(memoryId),
      ]);
    if (!currentVersion) return null;
    return { memory, currentVersion, versions, provenance, reviews, relations };
  }

  list(filter: MemoryListFilter): Promise<MemoryListPage> {
    return this.memories.list(filter);
  }

  async search(
    query: string,
    filter: Omit<MemoryListFilter, "cursor"> = {},
    limit = 50,
  ): Promise<SearchResultItem[]> {
    // Default to the same eligible-status set recall uses (Section 7.1):
    // rejected, superseded, and deprecated memories don't surface unless
    // the caller explicitly asks for them via `filter.statuses`.
    const statuses = filter.statuses ?? RECALL_ELIGIBLE_STATUSES;
    const candidates = await this.ftsSearch.search(query, Math.max(limit * 4, 100));
    const results: SearchResultItem[] = [];

    for (const candidate of candidates) {
      const memory = await this.memories.getById(candidate.memoryId);
      if (!memory) continue;
      if (!filter.includeDeleted && memory.status === "deleted") continue;
      if (!statuses.includes(memory.status)) continue;
      if (filter.kinds && !filter.kinds.includes(memory.kind)) continue;
      if (filter.scopes && !filter.scopes.some((s) => scopeMatches(s, memory.scope))) {
        continue;
      }
      const version = await this.memories.getVersion(candidate.memoryVersionId);
      if (!version) continue;
      results.push({ memory, version, rawScore: candidate.rawScore });
      if (results.length >= limit) break;
    }

    return results;
  }

  private async assertTransition(memoryId: string, to: MemoryStatus) {
    const memory = await this.memories.getById(memoryId);
    if (!memory) throw new Error(`Memory not found: ${memoryId}`);
    assertValidTransition(memory.status, to);
    return memory;
  }

  async approve(memoryId: string, actor: ActorRef, reason?: string): Promise<Memory> {
    const memory = await this.assertTransition(memoryId, "verified");
    const updated = await this.memories.updateStatus(memoryId, "verified");
    await this.reviews.create({
      memoryId,
      memoryVersionId: memory.currentVersionId,
      action: "approve",
      reason,
      actor,
    });
    return updated;
  }

  async reject(memoryId: string, actor: ActorRef, reason?: string): Promise<Memory> {
    const memory = await this.assertTransition(memoryId, "rejected");
    const updated = await this.memories.updateStatus(memoryId, "rejected");
    await this.reviews.create({
      memoryId,
      memoryVersionId: memory.currentVersionId,
      action: "reject",
      reason,
      actor,
    });
    return updated;
  }

  async deprecate(memoryId: string, actor: ActorRef, reason?: string): Promise<Memory> {
    const memory = await this.assertTransition(memoryId, "deprecated");
    const updated = await this.memories.updateStatus(memoryId, "deprecated");
    await this.reviews.create({
      memoryId,
      memoryVersionId: memory.currentVersionId,
      action: "deprecate",
      reason,
      actor,
    });
    return updated;
  }

  async restore(memoryId: string, actor: ActorRef, reason?: string): Promise<Memory> {
    const memory = await this.assertTransition(memoryId, "draft");
    const updated = await this.memories.updateStatus(memoryId, "draft");
    await this.reviews.create({
      memoryId,
      memoryVersionId: memory.currentVersionId,
      action: "restore",
      reason,
      actor,
    });
    return updated;
  }

  /** The old memory is marked superseded; a `supersedes` relation records the new memory as its replacement (Section 15.4). */
  async supersede(
    oldMemoryId: string,
    newMemoryId: string,
    actor: ActorRef,
    reason?: string,
  ): Promise<Memory> {
    const memory = await this.assertTransition(oldMemoryId, "superseded");
    const updated = await this.memories.updateStatus(oldMemoryId, "superseded", {
      supersededById: newMemoryId,
    });
    await this.reviews.create({
      memoryId: oldMemoryId,
      memoryVersionId: memory.currentVersionId,
      action: "supersede",
      reason,
      actor,
    });
    await this.relations.create({
      fromMemoryId: newMemoryId,
      toMemoryId: oldMemoryId,
      type: "supersedes",
      createdBy: actor,
    });
    return updated;
  }

  /** The source memory is folded into the target; deprecated rather than deleted so its history stays inspectable (Section 15.4). */
  async merge(
    sourceMemoryId: string,
    targetMemoryId: string,
    actor: ActorRef,
    reason?: string,
  ): Promise<Memory> {
    const memory = await this.assertTransition(sourceMemoryId, "deprecated");
    const updated = await this.memories.updateStatus(sourceMemoryId, "deprecated", {
      mergedIntoId: targetMemoryId,
    });
    await this.reviews.create({
      memoryId: sourceMemoryId,
      memoryVersionId: memory.currentVersionId,
      action: "merge",
      reason,
      actor,
    });
    await this.relations.create({
      fromMemoryId: sourceMemoryId,
      toMemoryId: targetMemoryId,
      type: "merged_into",
      createdBy: actor,
    });
    return updated;
  }

  /**
   * Tombstones the memory (ADR 0010): excluded from default list/search/
   * recall, but the row and its full history remain readable via
   * `includeDeleted`. Not logged as a ReviewAction — "delete" is not one of
   * the PRD's reviewable action types (Section 9.4), and mislabeling it as
   * e.g. "reject" would misrepresent the audit trail. The status change
   * itself (visible on the memory row) is the record of this action.
   */
  async delete(memoryId: string, _actor: ActorRef, _reason?: string): Promise<Memory> {
    await this.assertTransition(memoryId, "deleted");
    return this.memories.updateStatus(memoryId, "deleted");
  }

  addRelation(input: CreateMemoryRelationInput) {
    return this.relations.create(input);
  }
}

function scopeMatches(filterScope: ScopeRef, memoryScope: ScopeRef): boolean {
  return filterScope.level === memoryScope.level && filterScope.id === memoryScope.id;
}
