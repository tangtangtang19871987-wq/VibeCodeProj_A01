import type { ActorRef } from "./actors.js";

/** PRD Section 9.4 and 15.2. */
export const REVIEW_ACTIONS = [
  "approve",
  "edit",
  "reject",
  "merge",
  "supersede",
  "deprecate",
  "restore",
  "promote_scope",
] as const;
export type ReviewActionType = (typeof REVIEW_ACTIONS)[number];

export type ReviewAction = {
  id: string;
  memoryId: string;
  memoryVersionId: string;
  action: ReviewActionType;
  reason?: string;
  actor: ActorRef;
  createdAt: string;
};

/** PRD Section 15.2 consolidation outcomes, used by the Review Queue. */
export const CONSOLIDATION_OUTCOMES = [
  "ADD",
  "UPDATE_AS_NEW_VERSION",
  "MERGE",
  "SUPERSEDE",
  "REJECT",
  "IGNORE",
  "MARK_CONTRADICTION",
] as const;
export type ConsolidationOutcome = (typeof CONSOLIDATION_OUTCOMES)[number];

/** PRD Section 22.5 lineage relation types. */
export const MEMORY_RELATION_TYPES = [
  "derived_from",
  "supports",
  "contradicts",
  "supersedes",
  "merged_into",
] as const;
export type MemoryRelationType = (typeof MEMORY_RELATION_TYPES)[number];

export type MemoryRelation = {
  id: string;
  fromMemoryId: string;
  toMemoryId: string;
  type: MemoryRelationType;
  createdAt: string;
  createdBy: ActorRef;
  note?: string;
};
