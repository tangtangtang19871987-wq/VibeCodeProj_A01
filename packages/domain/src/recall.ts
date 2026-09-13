import type { MemoryKind, MemoryStatus } from "./memory.js";
import type { ScopeFilter, ScopeRef } from "./scope.js";

export type ContextBudget = {
  maxItems: number;
  maxChars: number;
};

/** PRD Section 9.6. A recall is a first-class, persisted domain object. */
export type RecallTrace = {
  id: string;
  sessionId: string;
  query: string;
  intent?: string;
  scopeFilter: ScopeFilter;
  kindFilter?: MemoryKind[];
  statusFilter: MemoryStatus[];
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

/** PRD Section 11.3 reason codes shown in the Retrieval Inspector. */
export const RECALL_REASON_CODES = [
  "MATCH_FTS_HIGH",
  "MATCH_FTS_MEDIUM",
  "MATCH_FTS_LOW",
  "SCOPE_PROJECT_EXACT",
  "SCOPE_AGENT_EXACT",
  "SCOPE_SESSION_EXACT",
  "SCOPE_GLOBAL_USER_ALLOWED",
  "STATUS_VERIFIED_BOOST",
  "STATUS_DRAFT_INCLUDED",
  "RECENCY_BOOST",
  "IMPORTANCE_BOOST",
  "FEEDBACK_POSITIVE_BOOST",
  "FEEDBACK_NEGATIVE_PENALTY",
  "STALE_PENALTY",
  "REJECTED_EXCLUDED",
  "DELETED_EXCLUDED",
  "SUPERSEDED_EXCLUDED",
  "OUT_OF_VALIDITY_WINDOW_EXCLUDED",
  "DRAFT_EXCLUDED_BY_POLICY",
  "OUT_OF_SCOPE_EXCLUDED",
  "BELOW_SCORE_THRESHOLD",
  "BUDGET_EXCLUDED",
  "DUPLICATE_EXCLUDED",
] as const;
export type RecallReasonCode = (typeof RECALL_REASON_CODES)[number];

/** Transparent, additive score components (Section 11.2, 11.3). */
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
  exclusionReasons: RecallReasonCode[];
  statusAtQueryTime: MemoryStatus;
  reasonCodes: RecallReasonCode[];
};

export type MemoryPacketItem = {
  id: string;
  kind: MemoryKind;
  content: string;
  scope: ScopeRef;
  status: MemoryStatus;
  why: string[];
  sourceHint?: string;
};

/** PRD Section 11.4 — the bounded response returned to a harness via MCP. */
export type MemoryPacket = {
  traceId: string;
  memories: MemoryPacketItem[];
  budget: {
    maxItems: number;
    returnedItems: number;
    estimatedTokens: number;
  };
};
