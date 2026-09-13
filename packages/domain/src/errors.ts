/**
 * Stable, machine-readable error codes surfaced over REST and MCP
 * (PRD Section 12.3, 18.3). Keep this list append-only in practice: removing
 * or renaming a code is a breaking change for clients.
 */
export const ERROR_CODES = [
  "NOT_FOUND",
  "INVALID_STATUS_TRANSITION",
  "VALIDATION_ERROR",
  "SCOPE_NOT_VISIBLE",
  "BUDGET_EXCEEDED",
  "LIMIT_EXCEEDED",
  "CONFLICT",
  "INTERNAL_ERROR",
] as const;
export type ErrorCode = (typeof ERROR_CODES)[number];

export class DomainError extends Error {
  constructor(
    public readonly code: ErrorCode,
    message: string,
    public readonly details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "DomainError";
  }
}

export class NotFoundError extends DomainError {
  constructor(entity: string, id: string) {
    super("NOT_FOUND", `${entity} not found: ${id}`, { entity, id });
    this.name = "NotFoundError";
  }
}
