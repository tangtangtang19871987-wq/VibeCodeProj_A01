import type {
  AgentSession,
  SessionEvent,
  SessionEventType,
  SessionOutcome,
  SessionStatus,
} from "@lams/domain";

export type OpenSessionInput = {
  externalSessionId?: string;
  harness: string;
  agentName?: string;
  projectId: string;
  title?: string;
  taskSummary?: string;
  metadata?: Record<string, unknown>;
};

export type UpdateSessionInput = {
  title?: string;
  taskSummary?: string;
  metadata?: Record<string, unknown>;
};

export type SessionListFilter = {
  projectId?: string;
  status?: SessionStatus[];
  limit?: number;
};

export interface SessionRepository {
  open(input: OpenSessionInput): Promise<AgentSession>;
  getById(id: string): Promise<AgentSession | null>;
  update(id: string, patch: UpdateSessionInput): Promise<AgentSession>;
  close(id: string, status: SessionStatus, outcome?: SessionOutcome): Promise<AgentSession>;
  list(filter: SessionListFilter): Promise<AgentSession[]>;

  appendEvent(
    sessionId: string,
    type: SessionEventType,
    payload: Record<string, unknown>,
  ): Promise<SessionEvent>;
  listEvents(sessionId: string): Promise<SessionEvent[]>;
}
