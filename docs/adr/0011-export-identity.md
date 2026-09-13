# ADR 0011: Export identity preservation vs. remapping

Status: Accepted

## Context

Section 25 item 8 asks whether export/import preserves original IDs or
deterministically remaps them on collision. Section 10.4 and Milestone 4
require import with dry-run and conflict reporting.

## Decision

Import always attempts to preserve original IDs (projects, memories,
versions, provenance, reviews, relations) so cross-instance references
stay stable. On a collision — an ID already present in the target
database — the importer does not overwrite silently:

- If the existing and incoming records are content-identical (same
  current version content hash), the import treats it as already-present
  and skips it (recorded as `skipped_identical` in the dry-run report).
- If they differ, the importer deterministically remaps the incoming
  record to a new ID (new ULID, so still time-sortable) and every
  reference to the old ID within the same import bundle is rewritten
  consistently. The conflict report lists old ID → new ID mappings and the
  reason (`skipped_identical` vs `remapped_conflict`) for every affected
  record, and nothing is written when run with `--dry-run`.

## Consequences

Two instances that have never diverged (e.g. a fresh clone) import with
identical IDs, making diffing and re-import idempotent. Instances that have
diverged never lose data to a silent overwrite; the user sees exactly what
was remapped before committing to a real import.

## Alternatives considered

Always remapping on import (never preserving IDs): rejected because it
would make repeated backup/restore cycles on the same machine needlessly
churn every ID, breaking external references (e.g. a bookmarked memory URL).
