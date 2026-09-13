export type ScopeFilter = {
  projectId: string;
  agentId?: string;
  sessionId?: string;
  allowedGlobalUser?: { level: string; id?: string }[];
};

export type ContextBudget = { maxItems: number; maxChars: number };

export type RecallTrace = {
  id: string;
  sessionId: string;
  query: string;
  intent?: string;
  scopeFilter: ScopeFilter;
  kindFilter?: string[];
  statusFilter: string[];
  strategy: string;
  strategyVersion: string;
  maxItems: number;
  budget: ContextBudget;
  candidateCount: number;
  selectedCount: number;
  returnedCount: number;
  returnedChars: number;
  estimatedTokens?: number;
  latencyMs: number;
  createdAt: string;
};

export type ScoreComponents = {
  ftsRelevance: number;
  scopeProximity: number;
  statusBoost: number;
  recencyBoost: number;
  importanceBoost: number;
  feedbackAdjustment: number;
  stalePenalty: number;
};

export type RecallCandidate = {
  id: string;
  recallTraceId: string;
  memoryId: string;
  memoryVersionId: string;
  eligible: boolean;
  rawScore: number;
  scoreComponents: ScoreComponents;
  finalScore: number;
  rank: number;
  selected: boolean;
  returned: boolean;
  exclusionReasons: string[];
  statusAtQueryTime: string;
  reasonCodes: string[];
  title?: string;
  content?: string;
};

export type DeliveryEvent = {
  id: string;
  sessionId: string;
  recallTraceId?: string;
  memoryId: string;
  memoryVersionId?: string;
  stage: "selected" | "returned" | "injected" | "used" | "outcome_linked";
  evidenceLevel: "observed" | "reported" | "inferred";
  note?: string;
  createdAt: string;
};

export type MemoryPacketItem = {
  id: string;
  kind: string;
  content: string;
  scope: { level: string; id?: string };
  status: string;
  why: string[];
  sourceHint?: string;
};

export type MemoryPacket = {
  traceId: string;
  memories: MemoryPacketItem[];
  budget: { maxItems: number; returnedItems: number; estimatedTokens: number };
};
