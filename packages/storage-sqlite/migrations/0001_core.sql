-- Milestone 0: core reference tables needed for a runnable health-checked
-- daemon. Memory/session/recall tables arrive in later numbered migrations
-- as their milestones are implemented (see docs/adr and PRD Section 10.1).

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  description TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  harness TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (name, harness)
);
