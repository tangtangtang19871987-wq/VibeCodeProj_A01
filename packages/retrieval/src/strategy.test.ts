import { describe, expect, it } from "vitest";
import { checkEligibility, scoreCandidate, selectWithinBudget } from "./strategy.js";

describe("checkEligibility", () => {
  const visibleScopes = [{ level: "project" as const, id: "p1" }];

  it("is eligible for a verified, in-scope memory with no validity window", () => {
    const result = checkEligibility(
      { status: "verified", scope: { level: "project", id: "p1" } },
      visibleScopes,
    );
    expect(result).toEqual({ eligible: true, reasons: [] });
  });

  it("excludes rejected memories", () => {
    const result = checkEligibility(
      { status: "rejected", scope: { level: "project", id: "p1" } },
      visibleScopes,
    );
    expect(result.eligible).toBe(false);
    expect(result.reasons).toContain("REJECTED_EXCLUDED");
  });

  it("excludes superseded and deprecated memories with the same reason code", () => {
    for (const status of ["superseded", "deprecated"] as const) {
      const result = checkEligibility({ status, scope: { level: "project", id: "p1" } }, visibleScopes);
      expect(result.eligible).toBe(false);
      expect(result.reasons).toContain("SUPERSEDED_EXCLUDED");
    }
  });

  it("excludes drafts unless includeDrafts is set", () => {
    const excluded = checkEligibility(
      { status: "draft", scope: { level: "project", id: "p1" } },
      visibleScopes,
      { includeDrafts: false },
    );
    expect(excluded.eligible).toBe(false);
    expect(excluded.reasons).toContain("DRAFT_EXCLUDED_BY_POLICY");

    const included = checkEligibility(
      { status: "draft", scope: { level: "project", id: "p1" } },
      visibleScopes,
      { includeDrafts: true },
    );
    expect(included.eligible).toBe(true);
  });

  it("excludes memories outside the visible scope set", () => {
    const result = checkEligibility(
      { status: "verified", scope: { level: "project", id: "other-project" } },
      visibleScopes,
    );
    expect(result.eligible).toBe(false);
    expect(result.reasons).toContain("OUT_OF_SCOPE_EXCLUDED");
  });

  it("excludes memories outside their validity window", () => {
    const asOf = new Date("2025-06-01T00:00:00Z");
    const expired = checkEligibility(
      { status: "verified", scope: { level: "project", id: "p1" }, validUntil: "2025-01-01T00:00:00Z" },
      visibleScopes,
      { includeDrafts: false, asOf },
    );
    expect(expired.eligible).toBe(false);
    expect(expired.reasons).toContain("OUT_OF_VALIDITY_WINDOW_EXCLUDED");

    const notYetValid = checkEligibility(
      { status: "verified", scope: { level: "project", id: "p1" }, validFrom: "2026-01-01T00:00:00Z" },
      visibleScopes,
      { includeDrafts: false, asOf },
    );
    expect(notYetValid.eligible).toBe(false);
    expect(notYetValid.reasons).toContain("OUT_OF_VALIDITY_WINDOW_EXCLUDED");
  });
});

describe("scoreCandidate", () => {
  const asOf = new Date("2025-06-01T00:00:00Z");

  it("scores a verified, exactly-scoped, recently-updated memory higher than a stale draft", () => {
    const strong = scoreCandidate(
      {
        status: "verified",
        scope: { level: "project", id: "p1" },
        importance: 0.8,
        updatedAt: "2025-05-30T00:00:00Z",
      },
      -5, // strong FTS match
      [{ level: "project", id: "p1" }],
      asOf,
    );
    const weak = scoreCandidate(
      {
        status: "draft",
        scope: { level: "project", id: "p1" },
        updatedAt: "2024-01-01T00:00:00Z", // stale
      },
      -0.1, // weak FTS match
      [{ level: "project", id: "p1" }],
      asOf,
    );

    expect(strong.finalScore).toBeGreaterThan(weak.finalScore);
    expect(strong.reasonCodes).toContain("STATUS_VERIFIED_BOOST");
    expect(strong.reasonCodes).toContain("SCOPE_PROJECT_EXACT");
    expect(strong.reasonCodes).toContain("IMPORTANCE_BOOST");
    expect(weak.reasonCodes).toContain("STALE_PENALTY");
  });

  it("flags a near-zero-relevance, unscoped, unboosted candidate as below threshold", () => {
    const result = scoreCandidate(
      { status: "draft", scope: { level: "global" }, updatedAt: "2025-05-31T00:00:00Z" },
      0, // no FTS relevance at all
      [], // not in any visible scope
      asOf,
    );
    expect(result.reasonCodes).toContain("BELOW_SCORE_THRESHOLD");
  });

  it("never produces a negative-infinity or NaN score", () => {
    const result = scoreCandidate(
      { status: "draft", scope: { level: "global" }, updatedAt: "2020-01-01T00:00:00Z" },
      0,
      [],
      asOf,
    );
    expect(Number.isFinite(result.finalScore)).toBe(true);
  });
});

describe("selectWithinBudget", () => {
  it("stops adding items once maxItems is reached", () => {
    const items = [1, 2, 3].map((n) => ({ item: n, chars: 10 }));
    const result = selectWithinBudget(items, { maxItems: 2, maxChars: 1000 });
    expect(result.selected).toEqual([1, 2]);
    expect(result.excludedForBudget).toEqual([3]);
  });

  it("stops adding items once maxChars would be exceeded, never exceeding it", () => {
    const items = [
      { item: "a", chars: 40 },
      { item: "b", chars: 40 },
      { item: "c", chars: 40 },
    ];
    const result = selectWithinBudget(items, { maxItems: 10, maxChars: 100 });
    expect(result.selected).toEqual(["a", "b"]);
    expect(result.excludedForBudget).toEqual(["c"]);
  });

  it("returns an empty selection when there are no candidates", () => {
    const result = selectWithinBudget([], { maxItems: 5, maxChars: 500 });
    expect(result.selected).toEqual([]);
    expect(result.excludedForBudget).toEqual([]);
  });
});
