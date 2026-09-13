import { monotonicFactory } from "ulidx";

const ulid = monotonicFactory();

/**
 * Entity ID prefixes. Prefixing IDs by kind makes a misrouted ID
 * (e.g. passing a session ID where a memory ID is expected) detectable
 * at a glance in logs, traces, and API payloads — see ADR 0003.
 */
export const ID_PREFIXES = {
  project: "proj",
  agent: "agent",
  session: "sess",
  event: "evt",
  memory: "mem",
  memoryVersion: "memv",
  provenance: "prov",
  review: "rev",
  relation: "rel",
  recallTrace: "rec",
  recallCandidate: "cand",
  deliveryEvent: "del",
  feedback: "fb",
  artifact: "art",
} as const;

export type EntityKind = keyof typeof ID_PREFIXES;

export function generateId(kind: EntityKind): string {
  return `${ID_PREFIXES[kind]}_${ulid()}`;
}

export function isIdOfKind(id: string, kind: EntityKind): boolean {
  return id.startsWith(`${ID_PREFIXES[kind]}_`);
}
