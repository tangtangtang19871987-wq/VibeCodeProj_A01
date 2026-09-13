import {
  checkEligibility,
  scoreCandidate,
  selectWithinBudget,
  STRATEGY_NAME,
  STRATEGY_VERSION,
} from "@lams/retrieval";
import {
  generateId,
  resolveVisibleScopes,
  type ContextBudget,
  type MemoryKind,
  type MemoryPacket,
  type MemoryPacketItem,
  type RecallCandidate,
  type RecallReasonCode,
  type RecallTrace,
  type ScopeFilter,
  type ScopeRef,
} from "@lams/domain";
import type { DeliveryRepository, RecallRepository } from "../ports/recall.js";
import type { FtsSearchPort, MemoryRepository } from "../ports/memory.js";
import type { ProvenanceRepository } from "../ports/provenance-review.js";
import type { SessionRepository } from "../ports/session.js";

export type RecallInput = {
  sessionId: string;
  query: string;
  intent?: string;
  kinds?: MemoryKind[];
  maxItems?: number;
  maxChars?: number;
  includeDrafts?: boolean;
  allowedGlobalUser?: ScopeRef[];
};

const DEFAULT_MAX_ITEMS = 5;
const DEFAULT_MAX_CHARS = 2000;
/**
 * How many raw FTS candidates to pull before filtering/scoring. Generous
 * relative to maxItems so eligibility/budget filtering has enough of a
 * pool to select from, while staying bounded (Section 18.3).
 */
const CANDIDATE_POOL_SIZE = 200;

export type RecallResult = {
  packet: MemoryPacket;
  trace: RecallTrace;
};

/**
 * The deterministic retrieval pipeline (Section 11.1): resolve scope,
 * filter, score, select under budget, persist the full trace, and return
 * only a compact packet. Every recall — successful or empty — persists a
 * trace; a recall is never returned without one (Milestone 2 acceptance
 * criteria).
 */
export class RetrievalService {
  constructor(
    private readonly memories: MemoryRepository,
    private readonly provenance: ProvenanceRepository,
    private readonly ftsSearch: FtsSearchPort,
    private readonly sessions: SessionRepository,
    private readonly recallRepo: RecallRepository,
    private readonly delivery: DeliveryRepository,
  ) {}

  async recall(input: RecallInput): Promise<RecallResult> {
    const startedAt = performance.now();
    const session = await this.sessions.getById(input.sessionId);
    if (!session) throw new Error(`Session not found: ${input.sessionId}`);

    const scopeFilter: ScopeFilter = {
      projectId: session.projectId,
      agentId: session.agentName,
      sessionId: session.id,
      allowedGlobalUser: input.allowedGlobalUser,
    };
    const visibleScopes = resolveVisibleScopes(scopeFilter);
    const includeDrafts = input.includeDrafts ?? false;
    const maxItems = input.maxItems ?? DEFAULT_MAX_ITEMS;
    const maxChars = input.maxChars ?? DEFAULT_MAX_CHARS;
    const budget: ContextBudget = { maxItems, maxChars };

    const ftsCandidates = await this.ftsSearch.search(input.query, CANDIDATE_POOL_SIZE);

    type Evaluated = {
      memoryId: string;
      memoryVersionId: string;
      rawScore: number;
      eligible: boolean;
      exclusionReasons: RecallReasonCode[];
      scoreComponents: RecallCandidate["scoreComponents"];
      finalScore: number;
      reasonCodes: RecallReasonCode[];
      statusAtQueryTime: RecallCandidate["statusAtQueryTime"];
      content: string;
      title: string;
      kind: MemoryKind;
      scope: ScopeRef;
    };

    const evaluated: Evaluated[] = [];
    const now = new Date();

    for (const candidate of ftsCandidates) {
      const memory = await this.memories.getById(candidate.memoryId);
      if (!memory) continue;
      // kind filtering happens before a candidate row even exists — it is
      // the caller narrowing the query's scope, not a policy exclusion
      // worth tracing (Section 11.1 step 2).
      if (input.kinds && !input.kinds.includes(memory.kind)) continue;

      const version = await this.memories.getVersion(candidate.memoryVersionId);
      if (!version) continue;

      const eligibility = checkEligibility(memory, visibleScopes, { includeDrafts, asOf: now });

      let scoreComponents: RecallCandidate["scoreComponents"] = {
        ftsRelevance: 0,
        scopeProximity: 0,
        statusBoost: 0,
        recencyBoost: 0,
        importanceBoost: 0,
        feedbackAdjustment: 0,
        stalePenalty: 0,
      };
      let finalScore = 0;
      let reasonCodes: RecallReasonCode[] = [];

      if (eligibility.eligible) {
        const scored = scoreCandidate(memory, candidate.rawScore, visibleScopes, now);
        scoreComponents = scored.scoreComponents;
        finalScore = scored.finalScore;
        reasonCodes = scored.reasonCodes;
      }

      evaluated.push({
        memoryId: memory.id,
        memoryVersionId: version.id,
        rawScore: candidate.rawScore,
        eligible: eligibility.eligible,
        exclusionReasons: eligibility.reasons,
        scoreComponents,
        finalScore,
        reasonCodes,
        statusAtQueryTime: memory.status,
        content: version.content,
        title: version.title,
        kind: memory.kind,
        scope: memory.scope,
      });
    }

    // Rank: eligible-and-above-threshold first (by score desc), then
    // everything else, so `rank` is meaningful for what could have been
    // selected while still covering every evaluated candidate.
    const isSelectable = (e: Evaluated) =>
      e.eligible && !e.reasonCodes.includes("BELOW_SCORE_THRESHOLD");
    const ranked = [...evaluated].sort((a, b) => {
      const aSelectable = isSelectable(a);
      const bSelectable = isSelectable(b);
      if (aSelectable !== bSelectable) return aSelectable ? -1 : 1;
      return b.finalScore - a.finalScore;
    });

    const selectableRanked = ranked.filter(isSelectable);
    const { selected } = selectWithinBudget(
      selectableRanked.map((e) => ({ item: e, chars: e.content.length })),
      budget,
    );
    const selectedIds = new Set(selected.map((e) => e.memoryVersionId));

    const traceId = generateId("recallTrace");
    const candidateRows: RecallCandidate[] = ranked.map((e, index) => {
      const isSelected = selectedIds.has(e.memoryVersionId);
      const budgetExcluded = isSelectable(e) && !isSelected;
      return {
        id: generateId("recallCandidate"),
        recallTraceId: traceId,
        memoryId: e.memoryId,
        memoryVersionId: e.memoryVersionId,
        eligible: e.eligible,
        rawScore: e.rawScore,
        scoreComponents: e.scoreComponents,
        finalScore: e.finalScore,
        rank: index + 1,
        selected: isSelected,
        returned: isSelected, // v1: no post-selection filtering, so returned === selected
        exclusionReasons: budgetExcluded
          ? [...e.exclusionReasons, "BUDGET_EXCLUDED"]
          : e.exclusionReasons,
        statusAtQueryTime: e.statusAtQueryTime,
        reasonCodes: e.reasonCodes,
      };
    });

    const returnedChars = selected.reduce((sum, e) => sum + e.content.length, 0);
    const latencyMs = performance.now() - startedAt;

    const trace: RecallTrace = {
      id: traceId,
      sessionId: session.id,
      query: input.query,
      intent: input.intent,
      scopeFilter,
      kindFilter: input.kinds,
      statusFilter: includeDrafts ? ["draft", "verified"] : ["verified"],
      strategy: STRATEGY_NAME,
      strategyVersion: STRATEGY_VERSION,
      maxItems,
      budget,
      candidateCount: evaluated.length,
      selectedCount: selected.length,
      returnedCount: selected.length,
      returnedChars,
      estimatedTokens: Math.ceil(returnedChars / 4), // ADR 0009: character-based budget is authoritative; this is a labeled estimate
      latencyMs,
      createdAt: new Date().toISOString(),
    };

    // Persist the trace and every candidate transactionally — a recall is
    // never returned without a durable, complete audit record.
    await this.recallRepo.createTraceWithCandidates(trace, candidateRows);

    await this.sessions.appendEvent(session.id, "recall_requested", {
      traceId,
      query: input.query,
      candidateCount: trace.candidateCount,
      returnedCount: trace.returnedCount,
    });

    for (const e of selected) {
      await this.delivery.record({
        sessionId: session.id,
        recallTraceId: traceId,
        memoryId: e.memoryId,
        memoryVersionId: e.memoryVersionId,
        stage: "selected",
        evidenceLevel: "observed",
      });
      await this.delivery.record({
        sessionId: session.id,
        recallTraceId: traceId,
        memoryId: e.memoryId,
        memoryVersionId: e.memoryVersionId,
        stage: "returned",
        evidenceLevel: "observed",
      });
    }

    const items: MemoryPacketItem[] = await Promise.all(
      selected.map(async (e) => {
        const provenanceList = await this.provenance.listByMemoryId(e.memoryId);
        return {
          id: e.memoryId,
          kind: e.kind,
          content: e.content,
          scope: e.scope,
          status: e.statusAtQueryTime,
          why: e.reasonCodes,
          sourceHint: provenanceList[0]?.sourceRef,
        };
      }),
    );

    const packet: MemoryPacket = {
      traceId,
      memories: items,
      budget: {
        maxItems,
        returnedItems: selected.length,
        estimatedTokens: trace.estimatedTokens ?? 0,
      },
    };

    return { packet, trace };
  }
}
