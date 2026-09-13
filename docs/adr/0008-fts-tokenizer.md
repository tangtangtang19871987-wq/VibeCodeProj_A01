# ADR 0008: FTS5 tokenizer

Status: Accepted

## Context

Section 25 item 5 leaves the FTS5 tokenizer choice open. We need one before
writing the `memory_fts` virtual table migration.

## Decision

Use the built-in `unicode61` tokenizer with `remove_diacritics 2`, applied
to a combined `title || ' ' || content || ' ' || summary || ' ' || tags`
column. This is English/Latin-script-oriented; non-Latin-script content
(e.g. CJK) will tokenize poorly (typically whole runs rather than words).

## Consequences

Retrieval quality for non-Latin-script memory content is a known, documented
limitation of v1, not a silent gap. Because retrieval strategy is versioned
(Section 11.2) and swappable (principle 4.9), a future ADR can introduce a
per-locale or trigram tokenizer as an additional, selectable strategy
without breaking existing recall traces, which record the strategy version
they used.

## Alternatives considered

`trigram` tokenizer: rejected as the default because it substantially
degrades exact-word relevance ranking for the common case (English
engineering notes) to gain multilingual substring matching that most v1
users will not need.
