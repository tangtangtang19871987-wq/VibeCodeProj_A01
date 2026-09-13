# ADR 0006: Evidence-level semantics

Status: Accepted

## Context

Section 9.7 and principle 4.7/4.9 require the system to never claim stronger
proof than it has for delivery stages (`selected`, `returned`, `injected`,
`used`, `outcome_linked`).

## Decision

Model `EvidenceLevel` as a closed enum: `observed | reported | inferred`,
attached to every `delivery_events` row, with these fixed rules enforced in
`packages/application` (not left to callers to get right):

- `selected` and `returned` rows are always written by the daemon itself
  during `memory_recall`, and are always `observed`.
- `injected` and `used` rows can only be written via explicit
  `memory_feedback` calls from a harness/agent, and are always `reported`
  unless the daemon itself performed the LLM call (not true for any v1
  harness), in which case they would be `observed` — this branch is
  currently unreachable and documented as such in code.
- `outcome_link` rows are `reported` when they come from harness feedback,
  or `inferred` when produced by a later analysis/evaluation step (Section
  17); the evaluation runner must set `inferred` explicitly and is never
  allowed to default to `observed` or `reported`.
- The UI (Retrieval Inspector, Session Inspector) always renders the
  evidence level as a visible label, never only as color, and the funnel
  view (Section 14.5) stops advancing past the last stage that has any
  recorded evidence.

## Consequences

It is structurally impossible for a `selected`/`returned` row to be
mislabeled `reported`, and impossible for `injected`/`used` to be silently
recorded as `observed`, because the write path for each stage is a distinct
application service method with a hardcoded evidence level, not a shared
method taking evidence level as a caller-supplied parameter.

## Alternatives considered

A single generic `record_delivery_event(stage, evidence_level)` method:
rejected because it would let a caller (including a future bug) assert
`observed` for a harness-reported event.
