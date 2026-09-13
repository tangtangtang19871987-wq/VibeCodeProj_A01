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
    throw new Error(`Request to ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<HealthReport>("/api/v1/health"),
  listProjects: () => request<{ projects: Project[] }>("/api/v1/projects"),
  createProject: (input: { key: string; name: string; description?: string }) =>
    request<{ project: Project }>("/api/v1/projects", {
      method: "POST",
      body: JSON.stringify(input),
    }),
};
