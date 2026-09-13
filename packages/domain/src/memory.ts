import type { ActorRef } from "./actors.js";
import type { ScopeRef } from "./scope.js";

/**
 * PRD Section 9.1. Deliberately excludes "skill" in v1 (Section 5.2, 9.1).
 */
export const MEMORY_KINDS = [
  "fact",
  "decision",
  "preference",
  "episode",
  "warning",
  "lesson",
] as const;
export type MemoryKind = (typeof MEMORY_KINDS)[number];

export const MEMORY_STATUSES = [
  "draft",
  "verified",
  "rejected",
  "superseded",
  "deprecated",
  "deleted",
] as const;
export type MemoryStatus = (typeof MEMORY_STATUSES)[number];

/** Statuses eligible for normal recall (Section 7.1, 11.1 step 5). */
export const RECALL_ELIGIBLE_STATUSES: readonly MemoryStatus[] = [
  "draft",
  "verified",
];

export type Memory = {
  id: string;
  kind: MemoryKind;
  status: MemoryStatus;
  scope: ScopeRef;
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

/**
 * Valid status transitions (Section 9.1, 15.2). Enforced centrally so no
 * call site can invent an undocumented lifecycle path.
 */
const ALLOWED_TRANSITIONS: Record<MemoryStatus, readonly MemoryStatus[]> = {
  draft: ["verified", "rejected", "deprecated", "deleted"],
  verified: ["superseded", "deprecated", "rejected", "deleted"],
  rejected: ["deleted", "draft"],
  superseded: ["deleted"],
  deprecated: ["deleted", "verified"],
  deleted: [],
};

export class InvalidStatusTransitionError extends Error {
  constructor(
    public readonly from: MemoryStatus,
    public readonly to: MemoryStatus,
  ) {
    super(`Cannot transition memory status from "${from}" to "${to}"`);
    this.name = "InvalidStatusTransitionError";
  }
}

export function canTransitionStatus(
  from: MemoryStatus,
  to: MemoryStatus,
): boolean {
  if (from === to) return true;
  return ALLOWED_TRANSITIONS[from].includes(to);
}

export function assertValidTransition(
  from: MemoryStatus,
  to: MemoryStatus,
): void {
  if (!canTransitionStatus(from, to)) {
    throw new InvalidStatusTransitionError(from, to);
  }
}

/**
 * Whether a memory is eligible for default recall given its status and
 * validity window at a given instant (Section 5.4, 7.1, 11.1).
 */
export function isRecallEligible(
  memory: Pick<Memory, "status" | "validFrom" | "validUntil">,
  options: { includeDrafts: boolean; asOf?: string } = { includeDrafts: false },
): boolean {
  if (memory.status === "verified") {
    // eligible, subject to validity window below
  } else if (memory.status === "draft") {
    if (!options.includeDrafts) return false;
  } else {
    return false;
  }
  const asOf = options.asOf ?? new Date().toISOString();
  if (memory.validFrom && memory.validFrom > asOf) return false;
  if (memory.validUntil && memory.validUntil < asOf) return false;
  return true;
}
