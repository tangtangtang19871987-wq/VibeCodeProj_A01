import type { AgentSession, SessionEvent, SessionOutcome, SessionStatus } from "@lams/domain";
import type {
  OpenSessionInput,
  SessionListFilter,
  SessionRepository,
  UpdateSessionInput,
} from "../ports/session.js";

/**
 * Session lifecycle orchestration (Section 9.5, 12.2 memory_session tool).
 * Every lifecycle transition also appends a session_event so the Session
 * Inspector timeline (Section 14.4) never has to reconstruct history from
 * anything but the append-only event log plus the session row itself.
 */
export class SessionService {
  constructor(private readonly sessions: SessionRepository) {}

  async open(input: OpenSessionInput): Promise<AgentSession> {
    const session = await this.sessions.open(input);
    await this.sessions.appendEvent(session.id, "session_opened", {
      harness: input.harness,
      agentName: input.agentName,
      projectId: input.projectId,
      taskSummary: input.taskSummary,
    });
    return session;
  }

  async update(id: string, patch: UpdateSessionInput): Promise<AgentSession> {
    const session = await this.sessions.update(id, patch);
    await this.sessions.appendEvent(id, "session_updated", { ...patch });
    return session;
  }

  async close(id: string, status: SessionStatus, outcome?: SessionOutcome): Promise<AgentSession> {
    const session = await this.sessions.close(id, status, outcome);
    await this.sessions.appendEvent(id, "session_closed", { status, outcome });
    return session;
  }

  getById(id: string): Promise<AgentSession | null> {
    return this.sessions.getById(id);
  }

  list(filter: SessionListFilter): Promise<AgentSession[]> {
    return this.sessions.list(filter);
  }

  listEvents(sessionId: string): Promise<SessionEvent[]> {
    return this.sessions.listEvents(sessionId);
  }
}
