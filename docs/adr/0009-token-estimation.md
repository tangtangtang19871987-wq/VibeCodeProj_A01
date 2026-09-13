# ADR 0009: Character-to-token estimation

Status: Accepted

## Context

Section 11.5 requires a hard character budget as the authoritative limit,
plus a labeled token estimate, since exact tokenizers vary by model
(Section 25 item 6).

## Decision

`estimated_tokens = ceil(returned_chars / 4)`, a widely-used rough English
heuristic. This value is always labeled `estimated` in the API response and
UI (never presented as an exact count), and it is never used to enforce the
budget — `max_chars` (a strict `returned_chars` limit) is the only
authoritative, enforced limit. The estimate is stored alongside the trace
for later comparison once real tokenizer adapters exist.

## Consequences

Token figures shown in the Dashboard and Retrieval Inspector are honestly
approximate. Because the actual limit is character-based, the system never
silently exceeds a caller's context budget due to tokenizer mismatch.

## Alternatives considered

Bundling a real tokenizer (e.g. `tiktoken`) per model: deferred — Section
11.5 explicitly allows "optional tokenizer adapters later," and pulling in
a model-specific tokenizer now would tie the budget system to one vendor's
encoding before any harness integration exists to justify the precision.
