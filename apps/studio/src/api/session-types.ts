export type SessionStatus = "open" | "completed" | "failed" | "abandoned";

export type SessionOutcome = {
  result: "success" | "failure" | "partial" | "unknown";
  summary?: string;
  metric?: Record<string, number>;
};

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

export type SessionEvent = {
  id: string;
  sessionId: string;
  type: string;
  payloadSchemaVersion: number;
  payload: Record<string, unknown>;
  createdAt: string;
};
