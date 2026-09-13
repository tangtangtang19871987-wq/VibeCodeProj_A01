import { describe, expect, it } from "vitest";
import {
  parseScope,
  resolveVisibleScopes,
  scopeToString,
  scopesEqual,
} from "./scope.js";

describe("scope parsing", () => {
  it("round-trips a project scope", () => {
    const s = parseScope("project/ilt-agent");
    expect(s).toEqual({ level: "project", id: "ilt-agent" });
    expect(scopeToString(s)).toBe("project/ilt-agent");
  });

  it("parses global scope without an id", () => {
    expect(parseScope("global")).toEqual({ level: "global", id: undefined });
  });

  it("rejects a non-global scope without an id", () => {
    expect(() => parseScope("project")).toThrow();
  });

  it("rejects an unknown level", () => {
    expect(() => parseScope("planet/earth")).toThrow();
  });
});

describe("resolveVisibleScopes", () => {
  it("always includes the project scope", () => {
    const scopes = resolveVisibleScopes({ projectId: "p1" });
    expect(scopes).toContainEqual({ level: "project", id: "p1" });
    expect(scopes).toHaveLength(1);
  });

  it("does not auto-include agent/session scope without being asked", () => {
    const scopes = resolveVisibleScopes({ projectId: "p1" });
    expect(scopes.some((s) => s.level === "agent")).toBe(false);
    expect(scopes.some((s) => s.level === "session")).toBe(false);
  });

  it("includes agent and session scope when explicitly named", () => {
    const scopes = resolveVisibleScopes({
      projectId: "p1",
      agentId: "a1",
      sessionId: "s1",
    });
    expect(scopes).toContainEqual({ level: "agent", id: "a1" });
    expect(scopes).toContainEqual({ level: "session", id: "s1" });
  });

  it("includes only explicitly allow-listed global/user scopes", () => {
    const scopes = resolveVisibleScopes({
      projectId: "p1",
      allowedGlobalUser: [{ level: "user", id: "local" }],
    });
    expect(scopes.some((s) => scopesEqual(s, { level: "user", id: "local" }))).toBe(
      true,
    );
  });

  it("rejects a non-global/user scope in allowedGlobalUser", () => {
    expect(() =>
      resolveVisibleScopes({
        projectId: "p1",
        allowedGlobalUser: [{ level: "project", id: "other" }],
      }),
    ).toThrow();
  });
});
