/**
 * Scope hierarchy (PRD Section 7): global < user < project < agent < session.
 * Scope is an organization/retrieval boundary in v1, not a hardened
 * multi-user authorization system (Section 7.2).
 */
export type ScopeLevel = "global" | "user" | "project" | "agent" | "session";

export const SCOPE_LEVEL_ORDER: readonly ScopeLevel[] = [
  "global",
  "user",
  "project",
  "agent",
  "session",
];

export type ScopeRef = {
  level: ScopeLevel;
  /** Required for every level except "global". */
  id?: string;
};

export function scopeToString(scope: ScopeRef): string {
  return scope.id ? `${scope.level}/${scope.id}` : scope.level;
}

export function parseScope(value: string): ScopeRef {
  const [level, ...rest] = value.split("/");
  if (!level || !isScopeLevel(level)) {
    throw new Error(`Invalid scope level in "${value}"`);
  }
  const id = rest.length > 0 ? rest.join("/") : undefined;
  if (level !== "global" && !id) {
    throw new Error(`Scope level "${level}" requires an id in "${value}"`);
  }
  return { level, id };
}

function isScopeLevel(value: string): value is ScopeLevel {
  return (SCOPE_LEVEL_ORDER as string[]).includes(value);
}

export type ScopeFilter = {
  /** The project the recalling session belongs to; always required. */
  projectId: string;
  /** The current agent, if the recall should include agent-scoped memory. */
  agentId?: string;
  /** The current session, if the recall should include session-scoped memory. */
  sessionId?: string;
  /**
   * Additional global/user scopes explicitly allowed for this recall.
   * Never inferred automatically (Section 7.1).
   */
  allowedGlobalUser?: ScopeRef[];
};

/**
 * Resolves which concrete scopes are visible for a recall, per Section 7.1:
 * project scope is always visible; agent/session scope only when the
 * recall names that agent/session; global/user scope only when explicitly
 * allow-listed. Promotion between scopes is never implicit.
 */
export function resolveVisibleScopes(filter: ScopeFilter): ScopeRef[] {
  const scopes: ScopeRef[] = [{ level: "project", id: filter.projectId }];
  if (filter.agentId) {
    scopes.push({ level: "agent", id: filter.agentId });
  }
  if (filter.sessionId) {
    scopes.push({ level: "session", id: filter.sessionId });
  }
  for (const scope of filter.allowedGlobalUser ?? []) {
    if (scope.level !== "global" && scope.level !== "user") {
      throw new Error(
        `allowedGlobalUser may only contain global/user scopes, got ${scopeToString(scope)}`,
      );
    }
    scopes.push(scope);
  }
  return scopes;
}

export function scopesEqual(a: ScopeRef, b: ScopeRef): boolean {
  return a.level === b.level && a.id === b.id;
}
