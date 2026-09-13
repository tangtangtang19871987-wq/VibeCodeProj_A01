export const SESSION_STATUSES = [
  "open",
  "completed",
  "failed",
  "abandoned",
] as const;
export type SessionStatus = (typeof SESSION_STATUSES)[number];

export type SessionOutcome = {
  result: "success" | "failure" | "partial" | "unknown";
  summary?: string;
  metric?: Record<string, number>;
};

/** PRD Section 9.5. */
export type AgentSession = {
  id: string;
  externalSessionId?: string;
  harness: string;
  agentName?: string;
  projectId: string;
  title?: string;
  taskSummary?: string;
  status: SessionStatus;
  startedAt: string;
  endedAt?: string;
  outcome?: SessionOutcome;
  metadata?: Record<string, unknown>;
};

/**
 * Append-only session event kinds recorded on the session timeline
 * (Section 9.5, 14.4). Kept as a closed enum + typed payload union so the
 * Session Inspector never has to interpret free-form JSON blindly.
 */
export const SESSION_EVENT_TYPES = [
  "session_opened",
  "session_updated",
  "session_closed",
  "recall_requested",
  "remember_proposed",
  "feedback_received",
  "human_annotation",
] as const;
export type SessionEventType = (typeof SESSION_EVENT_TYPES)[number];

export type SessionEvent = {
  id: string;
  sessionId: string;
  type: SessionEventType;
  /** Schema version of `payload`, so old events remain interpretable after payload shape changes. */
  payloadSchemaVersion: number;
  payload: Record<string, unknown>;
  createdAt: string;
};
