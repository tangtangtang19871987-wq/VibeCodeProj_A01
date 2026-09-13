import type {
  ActorRef,
  Memory,
  MemoryKind,
  MemoryStatus,
  MemoryVersion,
  ScopeRef,
} from "@lams/domain";

export type CreateMemoryInput = {
  kind: MemoryKind;
  scope: ScopeRef;
  status?: MemoryStatus;
  confidence?: number;
  importance?: number;
  validFrom?: string;
  validUntil?: string;
  createdBy: ActorRef;
  version: {
    title: string;
    content: string;
    summary?: string;
    tags?: string[];
    structuredData?: Record<string, unknown>;
    changeReason?: string;
  };
};

export type AddVersionInput = {
  title: string;
  content: string;
  summary?: string;
  tags?: string[];
  structuredData?: Record<string, unknown>;
  changeReason?: string;
  createdBy: ActorRef;
};

export type MemoryListFilter = {
  scopes?: ScopeRef[];
  statuses?: MemoryStatus[];
  kinds?: MemoryKind[];
  includeDeleted?: boolean;
  limit?: number;
  cursor?: string;
};

export type MemoryListPage = {
  items: Array<{ memory: Memory; currentVersion: MemoryVersion }>;
  nextCursor?: string;
};

export interface MemoryRepository {
  createWithFirstVersion(
    input: CreateMemoryInput,
  ): Promise<{ memory: Memory; version: MemoryVersion }>;
  getById(id: string): Promise<Memory | null>;
  getCurrentVersion(memoryId: string): Promise<MemoryVersion | null>;
  getVersion(versionId: string): Promise<MemoryVersion | null>;
  listVersions(memoryId: string): Promise<MemoryVersion[]>;
  addVersion(
    memoryId: string,
    input: AddVersionInput,
  ): Promise<{ memory: Memory; version: MemoryVersion }>;
  updateStatus(
    memoryId: string,
    status: MemoryStatus,
    extra?: { supersededById?: string; mergedIntoId?: string },
  ): Promise<Memory>;
  list(filter: MemoryListFilter): Promise<MemoryListPage>;
}

export type SearchCandidate = {
  memoryId: string;
  memoryVersionId: string;
  rawScore: number;
};

export interface FtsSearchPort {
  /**
   * Full-text search over current memory versions. Returns raw BM25-style
   * scores (more negative = more relevant, per SQLite FTS5 convention) for
   * the retrieval service to normalize (Section 11.2).
   */
  search(query: string, limit: number): Promise<SearchCandidate[]>;
}
