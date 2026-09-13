/**
 * PRD Section 9.7. See ADR 0006 for the enforcement rules that keep evidence
 * levels honest — this module only defines the closed vocabulary and the
 * fixed stage -> allowed-evidence-level mapping; application services must
 * not let callers pick an evidence level outside what a stage allows.
 */
export const DELIVERY_STAGES = [
  "selected",
  "returned",
  "injected",
  "used",
  "outcome_linked",
] as const;
export type DeliveryStage = (typeof DELIVERY_STAGES)[number];

export const EVIDENCE_LEVELS = ["observed", "reported", "inferred"] as const;
export type EvidenceLevel = (typeof EVIDENCE_LEVELS)[number];

/**
 * Which evidence levels are legal for each stage. `selected`/`returned` are
 * always daemon-observed; `injected`/`used`/`outcome_linked` can only be
 * reported or inferred, never observed, because no v1 harness lets the
 * daemon see the actual model request (ADR 0006).
 */
export const ALLOWED_EVIDENCE_LEVELS: Record<
  DeliveryStage,
  readonly EvidenceLevel[]
> = {
  selected: ["observed"],
  returned: ["observed"],
  injected: ["reported"],
  used: ["reported", "inferred"],
  outcome_linked: ["reported", "inferred"],
};

export function isEvidenceLevelAllowed(
  stage: DeliveryStage,
  level: EvidenceLevel,
): boolean {
  return ALLOWED_EVIDENCE_LEVELS[stage].includes(level);
}

export type DeliveryEvent = {
  id: string;
  sessionId: string;
  recallTraceId?: string;
  memoryId: string;
  memoryVersionId?: string;
  stage: DeliveryStage;
  evidenceLevel: EvidenceLevel;
  note?: string;
  createdAt: string;
};

export const FEEDBACK_EVENT_TYPES = [
  "injected",
  "useful",
  "not_useful",
  "harmful",
  "stale",
  "incorrect",
  "outcome_link",
] as const;
export type FeedbackEventType = (typeof FEEDBACK_EVENT_TYPES)[number];

export type MemoryFeedback = {
  id: string;
  sessionId: string;
  recallTraceId?: string;
  memoryId: string;
  eventType: FeedbackEventType;
  evidenceLevel: EvidenceLevel;
  note?: string;
  outcomeMetric?: Record<string, number>;
  createdAt: string;
};
