# ADR 0010: Tombstoned content visibility

Status: Accepted

## Context

Section 25 item 7 asks whether tombstoned content stays readable to
administrators in v1. Section 10.2 defaults deletion to tombstoning so
audits are not broken.

## Decision

A `deleted` memory is excluded from every default list/search/recall
result, but remains fully readable — content, versions, provenance, review
history — via `GET /api/v1/memories/:id?include_deleted=true` and an
explicit "Show deleted" toggle in the Memory Explorer (never on by default).
Since v1 is single-user local (Section 4.10, 7.2), "administrator" and "the
user" are the same actor, so no separate role check is introduced. A future
physical purge operation (Section 10.2) is a distinct, explicitly
irreversible action, out of scope for v1.

## Consequences

Deleting a memory never destroys audit trail or breaks historical recall
reconstruction (Section 15.4), while normal browsing/search/recall never
surfaces deleted content by accident.

## Alternatives considered

Making tombstoned content unreadable through the ordinary API (only visible
via raw DB inspection): rejected because it would make Section 10.2's audit
guarantee unverifiable from the product surface itself.
