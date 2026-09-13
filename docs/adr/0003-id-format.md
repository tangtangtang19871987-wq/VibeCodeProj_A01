# ADR 0003: ID format — ULID via `ulidx`

Status: Accepted

## Context

Every domain entity needs a globally sortable, collision-resistant ID. The
PRD's example IDs (`mem_01...`, `rec_01...`) are prefixed ULIDs.

## Decision

Use the `ulidx` package (a maintained, monotonic-capable ULID implementation)
to generate 26-character Crockford base32 ULIDs, prefixed per entity kind
(`mem_`, `memv_`, `prov_`, `rev_`, `proj_`, `sess_`, `evt_`, `rec_`, `cand_`,
`del_`, `fb_`, `art_`, `agent_`). Use the monotonic factory within a single
process so IDs generated in the same millisecond still sort correctly, which
matters for reconstructing append-only timelines (session events, recall
candidates) in creation order without a separate sequence column.

## Consequences

IDs are lexically sortable by creation time, human-scannable enough to
paste into a UI or MCP payload, and prefix-typed so a stray ID cannot be
silently used against the wrong table/repository.

## Alternatives considered

UUIDv4: rejected because it is not time-sortable, which would force a
separate `created_at` sort column everywhere ordering matters.
Auto-increment integers: rejected because they leak table cardinality and
complicate the export/import identity story (ADR 0011).
