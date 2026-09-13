export const PROVENANCE_SOURCE_TYPES = [
  "session_event",
  "tool_result",
  "artifact",
  "human",
  "import",
  "memory",
] as const;
export type ProvenanceSourceType = (typeof PROVENANCE_SOURCE_TYPES)[number];

/** PRD Section 9.3. */
export type Provenance = {
  id: string;
  memoryId: string;
  memoryVersionId?: string;
  sourceType: ProvenanceSourceType;
  sourceRef: string;
  sessionId?: string;
  eventId?: string;
  excerpt?: string;
  artifactHash?: string;
  createdAt: string;
};
