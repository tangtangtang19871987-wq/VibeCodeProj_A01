export type MemoryKind = "fact" | "decision" | "preference" | "episode" | "warning" | "lesson";
export type MemoryStatus = "draft" | "verified" | "rejected" | "superseded" | "deprecated" | "deleted";
export type ScopeLevel = "global" | "user" | "project" | "agent" | "session";

export type ScopeRef = { level: ScopeLevel; id?: string };

export type ActorRef = { type: "human" | "agent" | "system" | "import"; id: string; displayName?: string };

export type Memory = {
  id: string;
  kind: MemoryKind;
  status: MemoryStatus;
  scope: ScopeRef;
  scopeDisplay: string;
  currentVersionId: string;
  confidence?: number;
  importance?: number;
  validFrom?: string;
  validUntil?: string;
  createdAt: string;
  createdBy: ActorRef;
  updatedAt: string;
  supersededById?: string;
  mergedIntoId?: string;
};

export type MemoryVersion = {
  id: string;
  memoryId: string;
  version: number;
  title: string;
  content: string;
  summary?: string;
  tags: string[];
  structuredData?: Record<string, unknown>;
  changeReason?: string;
  createdAt: string;
  createdBy: ActorRef;
};

export type Provenance = {
  id: string;
  memoryId: string;
  memoryVersionId?: string;
  sourceType: string;
  sourceRef: string;
  sessionId?: string;
  eventId?: string;
  excerpt?: string;
  artifactHash?: string;
  createdAt: string;
};

export type ReviewAction = {
  id: string;
  memoryId: string;
  memoryVersionId: string;
  action: string;
  reason?: string;
  actor: ActorRef;
  createdAt: string;
};

export type MemoryRelation = {
  id: string;
  fromMemoryId: string;
  toMemoryId: string;
  type: string;
  note?: string;
  createdAt: string;
  createdBy: ActorRef;
};

export type MemoryListItem = { memory: Memory; currentVersion: MemoryVersion; rawScore?: number };

export type MemoryDetail = {
  memory: Memory;
  currentVersion: MemoryVersion;
  versions: MemoryVersion[];
  provenance: Provenance[];
  reviews: ReviewAction[];
  relations: MemoryRelation[];
};
