import type { MemoryDetail, MemoryKind, MemoryListItem, MemoryStatus } from "./memory-types.js";
import type { AgentSession, SessionEvent, SessionStatus } from "./session-types.js";
import type { DeliveryEvent, MemoryPacket, RecallCandidate, RecallTrace } from "./recall-types.js";

export * from "./memory-types.js";
export * from "./session-types.js";
export * from "./recall-types.js";

export type HealthReport = {
  ok: boolean;
  checks: {
    process: { ok: boolean };
    database: { ok: boolean; detail?: string };
    migrations: {
      ok: boolean;
      detail?: string;
      appliedMigrations: string[];
      pendingMigrations: string[];
    };
    fts: { ok: boolean; detail?: string };
    artifactDirectory: { ok: boolean; detail?: string };
  };
};

export type Project = {
  id: string;
  key: string;
  name: string;
  description?: string;
  createdAt: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok && res.status !== 503) {
    const body = await res.text();
    throw new Error(`Request to ${path} failed: ${res.status} ${res.statusText} — ${body}`);
  }
  return (await res.json()) as T;
}

export type MemoryListParams = {
  q?: string;
  status?: MemoryStatus[];
  kind?: MemoryKind[];
  scope?: string;
  includeDeleted?: boolean;
};

function toQueryString(params: MemoryListParams): string {
  const sp = new URLSearchParams();
  if (params.q) sp.set("q", params.q);
  if (params.status?.length) sp.set("status", params.status.join(","));
  if (params.kind?.length) sp.set("kind", params.kind.join(","));
  if (params.scope) sp.set("scope", params.scope);
  if (params.includeDeleted) sp.set("includeDeleted", "true");
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export type CreateMemoryInput = {
  kind: MemoryKind;
  scope: { level: string; id?: string };
  status?: MemoryStatus;
  version: { title: string; content: string; summary?: string; tags?: string[] };
};

export const api = {
  health: () => request<HealthReport>("/api/v1/health"),
  listProjects: () => request<{ projects: Project[] }>("/api/v1/projects"),
  createProject: (input: { key: string; name: string; description?: string }) =>
    request<{ project: Project }>("/api/v1/projects", {
      method: "POST",
      body: JSON.stringify(input),
    }),

  listMemories: (params: MemoryListParams) =>
    request<{ items: MemoryListItem[]; nextCursor?: string }>(
      `/api/v1/memories${toQueryString(params)}`,
    ),
  getMemory: (id: string) => request<MemoryDetail>(`/api/v1/memories/${id}`),
  createMemory: (input: CreateMemoryInput) =>
    request<{ memory: unknown; version: unknown }>("/api/v1/memories", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  addVersion: (
    id: string,
    input: { title: string; content: string; summary?: string; tags?: string[]; changeReason?: string },
  ) =>
    request(`/api/v1/memories/${id}/versions`, {
      method: "POST",
      body: JSON.stringify(input),
    }),
  reviewAction: (
    id: string,
    input: { action: string; reason?: string; targetMemoryId?: string },
  ) =>
    request(`/api/v1/memories/${id}/reviews`, {
      method: "POST",
      body: JSON.stringify(input),
    }),
  deleteMemory: (id: string) => request(`/api/v1/memories/${id}`, { method: "DELETE" }),

  listSessions: (params: { projectId?: string; status?: SessionStatus[] } = {}) => {
    const sp = new URLSearchParams();
    if (params.projectId) sp.set("projectId", params.projectId);
    if (params.status?.length) sp.set("status", params.status.join(","));
    const qs = sp.toString();
    return request<{ sessions: AgentSession[] }>(`/api/v1/sessions${qs ? `?${qs}` : ""}`);
  },
  getSession: (id: string) =>
    request<{ session: AgentSession; events: SessionEvent[] }>(`/api/v1/sessions/${id}`),
  openSession: (input: {
    harness: string;
    projectId: string;
    agentName?: string;
    taskSummary?: string;
  }) =>
    request<{ session: AgentSession }>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify(input),
    }),

  recall: (input: {
    sessionId: string;
    query: string;
    intent?: string;
    kinds?: MemoryKind[];
    maxItems?: number;
    maxChars?: number;
    includeDrafts?: boolean;
  }) =>
    request<{ packet: MemoryPacket; trace: RecallTrace }>("/api/v1/recalls", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  getRecallCandidates: (traceId: string) =>
    request<{ trace: RecallTrace; candidates: RecallCandidate[]; deliveryEvents: DeliveryEvent[] }>(
      `/api/v1/recalls/${traceId}/candidates`,
    ),
};
