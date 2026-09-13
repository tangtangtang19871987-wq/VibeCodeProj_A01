# ADR 0002: Database access layer — Kysely over Drizzle

Status: Accepted

## Context

Section 25.1 leaves the choice between Kysely and Drizzle open. Both support
SQLite and TypeScript type inference. We need one answer before writing the
first migration.

## Decision

Use Kysely as the typed query builder, with a small hand-rolled migration
runner (numbered `.sql` files applied in order, tracked in
`schema_migrations`), rather than Drizzle's schema-first DSL and migration
generator.

Reasons:

- Kysely is a pure query builder with no required schema DSL: the domain
  package's hand-written types stay the single source of truth for shape,
  and Kysely's `Database` interface is a thin, explicit mapping onto SQL
  tables. This keeps the storage layer honest about what SQL actually runs,
  which matters for an audit-focused product.
- Numbered raw-SQL migrations are easier to read and diff than a generated
  migration DSL, and they make FTS5 virtual table / trigger definitions
  (which Drizzle does not model well) straightforward to express.
- Kysely's plain `better-sqlite3` dialect has no native/runtime binding
  surprises.

## Consequences

We write SQL migrations by hand and maintain a parallel `Database` interface
for Kysely. This is more manual than Drizzle's codegen, but keeps the schema
legible and avoids fighting an ORM's opinions about versioning, soft deletes,
and virtual tables — all of which this product needs to control precisely.

## Alternatives considered

Drizzle ORM: rejected for v1 because its migration generator does not
cleanly express FTS5 virtual tables and triggers, which are central to the
retrieval pipeline.
