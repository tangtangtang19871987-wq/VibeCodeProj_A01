# ADR 0012: `memory_remember` accepts multiple candidates per call

Status: Accepted

## Context

Section 25 item 9 asks whether `memory_remember` should accept one or many
candidates per call.

## Decision

`memory_remember` accepts an array of 1–5 candidates in one call (hard cap
enforced server-side, Section 18.3). A single agent turn commonly surfaces
more than one worth-remembering fact at once (e.g. a decision and the
warning that motivated it), and forcing one call per candidate would
multiply MCP round-trips without reducing schema complexity — the per-
candidate shape (`kind`, `title`, `content`, `scope`, `confidence`, `tags`,
`evidence`) is identical whether sent one at a time or batched.

## Consequences

The tool schema has one extra array wrapper but no added conceptual
complexity. The 5-candidate cap keeps a single call bounded, consistent
with Section 18.3's input-limit requirement and the "context-efficient MCP"
principle (4.5).

## Alternatives considered

Strictly one candidate per call: rejected as it would not reduce schema
complexity (per Section 25 item 9's own framing) while adding round-trips
for the common multi-fact case.
