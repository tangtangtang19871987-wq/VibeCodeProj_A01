import { describe, expect, it } from "vitest";
import {
  InvalidStatusTransitionError,
  assertValidTransition,
  canTransitionStatus,
  isRecallEligible,
} from "./memory.js";

describe("memory status transitions", () => {
  it("allows draft -> verified", () => {
    expect(canTransitionStatus("draft", "verified")).toBe(true);
  });

  it("allows verified -> superseded", () => {
    expect(canTransitionStatus("verified", "superseded")).toBe(true);
  });

  it("rejects superseded -> verified", () => {
    expect(canTransitionStatus("superseded", "verified")).toBe(false);
    expect(() => assertValidTransition("superseded", "verified")).toThrow(
      InvalidStatusTransitionError,
    );
  });

  it("rejects deleted -> anything", () => {
    expect(canTransitionStatus("deleted", "draft")).toBe(false);
  });

  it("is a no-op for identical from/to", () => {
    expect(canTransitionStatus("verified", "verified")).toBe(true);
  });
});

describe("recall eligibility", () => {
  it("excludes drafts by default", () => {
    expect(
      isRecallEligible({ status: "draft" }, { includeDrafts: false }),
    ).toBe(false);
  });

  it("includes drafts when explicitly requested", () => {
    expect(
      isRecallEligible({ status: "draft" }, { includeDrafts: true }),
    ).toBe(true);
  });

  it("excludes superseded and rejected regardless of includeDrafts", () => {
    expect(
      isRecallEligible({ status: "superseded" }, { includeDrafts: true }),
    ).toBe(false);
    expect(
      isRecallEligible({ status: "rejected" }, { includeDrafts: true }),
    ).toBe(false);
  });

  it("excludes verified memory outside its validity window", () => {
    expect(
      isRecallEligible(
        { status: "verified", validUntil: "2020-01-01T00:00:00Z" },
        { includeDrafts: false, asOf: "2021-01-01T00:00:00Z" },
      ),
    ).toBe(false);
  });

  it("includes verified memory inside its validity window", () => {
    expect(
      isRecallEligible(
        {
          status: "verified",
          validFrom: "2020-01-01T00:00:00Z",
          validUntil: "2030-01-01T00:00:00Z",
        },
        { includeDrafts: false, asOf: "2021-01-01T00:00:00Z" },
      ),
    ).toBe(true);
  });
});
