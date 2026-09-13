import {
  scopesEqual,
  type Memory,
  type MemoryStatus,
  type RecallReasonCode,
  type ScopeRef,
  type ScoreComponents,
} from "@lams/domain";

/**
 * PRD Section 11.2/11.3: the retrieval strategy is named and versioned so
 * every recall trace records exactly which deterministic formula produced
 * it, and a future strategy can be introduced without invalidating old
 * traces. This is intentionally simple — a documented heuristic, not a
 * claim of optimality (Section 11.2).
 */
export const STRATEGY_NAME = "deterministic-fts-v1";
export const STRATEGY_VERSION = "1.0.0";

export const SCORE_WEIGHTS = {
  ftsRelevance: 0.5,
  scopeProximity: 0.15,
  statusBoost: 0.15,
  recencyBoost: 0.1,
  importanceBoost: 0.1,
} as const;

export const MIN_SCORE_THRESHOLD = 0.2;
const RECENCY_HALF_LIFE_DAYS = 90;
const STALE_AFTER_DAYS = 180;
const STALE_PENALTY_AMOUNT = 0.1;

/** SQLite FTS5's bm25() is unbounded and more-negative-is-better; map it onto [0, 1]. */
export function normalizeFtsScore(rawScore: number): number {
  const magnitude = Math.max(0, -rawScore);
  return magnitude / (1 + magnitude);
}

function classifyMatch(normalized: number): RecallReasonCode {
  if (normalized >= 0.6) return "MATCH_FTS_HIGH";
  if (normalized >= 0.3) return "MATCH_FTS_MEDIUM";
  return "MATCH_FTS_LOW";
}

/**
 * Scope proximity: the most specific scope the memory matches wins. Order
 * mirrors PRD Section 7 (session > agent > project > global/user).
 */
function scopeProximity(
  memoryScope: ScopeRef,
  visibleScopes: ScopeRef[],
): { value: number; reasonCode: RecallReasonCode } | null {
  const match = visibleScopes.find((s) => scopesEqual(s, memoryScope));
  if (!match) return null;
  switch (match.level) {
    case "session":
      return { value: 1.0, reasonCode: "SCOPE_SESSION_EXACT" };
    case "agent":
      return { value: 0.8, reasonCode: "SCOPE_AGENT_EXACT" };
    case "project":
      return { value: 0.6, reasonCode: "SCOPE_PROJECT_EXACT" };
    default:
      return { value: 0.4, reasonCode: "SCOPE_GLOBAL_USER_ALLOWED" };
  }
}

function statusBoost(status: MemoryStatus): { value: number; reasonCode: RecallReasonCode } {
  return status === "verified"
    ? { value: 1.0, reasonCode: "STATUS_VERIFIED_BOOST" }
    : { value: 0.5, reasonCode: "STATUS_DRAFT_INCLUDED" };
}

function recencyBoost(updatedAt: string, asOf: Date): number {
  const ageDays = (asOf.getTime() - new Date(updatedAt).getTime()) / 86_400_000;
  return Math.exp(-Math.max(0, ageDays) / RECENCY_HALF_LIFE_DAYS);
}

function stalePenalty(updatedAt: string, asOf: Date): number {
  const ageDays = (asOf.getTime() - new Date(updatedAt).getTime()) / 86_400_000;
  return ageDays > STALE_AFTER_DAYS ? -STALE_PENALTY_AMOUNT : 0;
}

export type EligibilityInput = {
  status: MemoryStatus;
  scope: ScopeRef;
  validFrom?: string;
  validUntil?: string;
};

export type EligibilityResult = {
  eligible: boolean;
  reasons: RecallReasonCode[];
};

/**
 * Pipeline steps 2 and 5 (Section 11.1): status/scope/validity eligibility,
 * independent of ranking. `deprecated` is grouped with `superseded` for
 * exclusion purposes — both mean "do not use this directly" even though
 * they are distinct statuses (Section 9.1).
 */
export function checkEligibility(
  memory: EligibilityInput,
  visibleScopes: ScopeRef[],
  options: { includeDrafts: boolean; asOf?: Date } = { includeDrafts: false },
): EligibilityResult {
  const reasons: RecallReasonCode[] = [];

  if (memory.status === "deleted") reasons.push("DELETED_EXCLUDED");
  if (memory.status === "rejected") reasons.push("REJECTED_EXCLUDED");
  if (memory.status === "superseded" || memory.status === "deprecated") {
    reasons.push("SUPERSEDED_EXCLUDED");
  }
  if (memory.status === "draft" && !options.includeDrafts) {
    reasons.push("DRAFT_EXCLUDED_BY_POLICY");
  }
  if (!visibleScopes.some((s) => scopesEqual(s, memory.scope))) {
    reasons.push("OUT_OF_SCOPE_EXCLUDED");
  }

  const asOf = (options.asOf ?? new Date()).toISOString();
  if (memory.validFrom && memory.validFrom > asOf) {
    reasons.push("OUT_OF_VALIDITY_WINDOW_EXCLUDED");
  }
  if (memory.validUntil && memory.validUntil < asOf) {
    reasons.push("OUT_OF_VALIDITY_WINDOW_EXCLUDED");
  }

  return { eligible: reasons.length === 0, reasons };
}

export type ScoredCandidate = {
  scoreComponents: ScoreComponents;
  finalScore: number;
  reasonCodes: RecallReasonCode[];
};

/** Pipeline step 4 (Section 11.2): transparent, additive scoring for an eligible candidate. */
export function scoreCandidate(
  memory: Pick<Memory, "status" | "scope" | "importance" | "updatedAt">,
  rawFtsScore: number,
  visibleScopes: ScopeRef[],
  asOf: Date = new Date(),
): ScoredCandidate {
  const reasonCodes: RecallReasonCode[] = [];

  const normalizedFts = normalizeFtsScore(rawFtsScore);
  reasonCodes.push(classifyMatch(normalizedFts));

  const scope = scopeProximity(memory.scope, visibleScopes);
  if (scope) reasonCodes.push(scope.reasonCode);

  const status = statusBoost(memory.status);
  reasonCodes.push(status.reasonCode);

  const recency = recencyBoost(memory.updatedAt, asOf);
  if (recency >= 0.5) reasonCodes.push("RECENCY_BOOST");

  const importance = memory.importance ?? 0;
  if (importance >= 0.5) reasonCodes.push("IMPORTANCE_BOOST");

  const stale = stalePenalty(memory.updatedAt, asOf);
  if (stale < 0) reasonCodes.push("STALE_PENALTY");

  const scoreComponents: ScoreComponents = {
    ftsRelevance: normalizedFts,
    scopeProximity: scope?.value ?? 0,
    statusBoost: status.value,
    recencyBoost: recency,
    importanceBoost: importance,
    feedbackAdjustment: 0, // populated once Milestone 5's feedback aggregates exist
    stalePenalty: stale,
  };

  const finalScore =
    SCORE_WEIGHTS.ftsRelevance * scoreComponents.ftsRelevance +
    SCORE_WEIGHTS.scopeProximity * scoreComponents.scopeProximity +
    SCORE_WEIGHTS.statusBoost * scoreComponents.statusBoost +
    SCORE_WEIGHTS.recencyBoost * scoreComponents.recencyBoost +
    SCORE_WEIGHTS.importanceBoost * scoreComponents.importanceBoost +
    scoreComponents.feedbackAdjustment +
    scoreComponents.stalePenalty;

  if (finalScore < MIN_SCORE_THRESHOLD) reasonCodes.push("BELOW_SCORE_THRESHOLD");

  return { scoreComponents, finalScore, reasonCodes };
}

export type BudgetItem<T> = {
  item: T;
  chars: number;
};

export type BudgetSelection<T> = {
  selected: T[];
  excludedForBudget: T[];
};

/** Pipeline step 7 (Section 11.1, 11.5): greedy selection under a hard character budget and item count. */
export function selectWithinBudget<T>(
  rankedEligible: BudgetItem<T>[],
  budget: { maxItems: number; maxChars: number },
): BudgetSelection<T> {
  const selected: T[] = [];
  const excludedForBudget: T[] = [];
  let usedChars = 0;

  for (const candidate of rankedEligible) {
    const wouldExceedItems = selected.length >= budget.maxItems;
    const wouldExceedChars = usedChars + candidate.chars > budget.maxChars;
    if (wouldExceedItems || wouldExceedChars) {
      excludedForBudget.push(candidate.item);
      continue;
    }
    selected.push(candidate.item);
    usedChars += candidate.chars;
  }

  return { selected, excludedForBudget };
}
