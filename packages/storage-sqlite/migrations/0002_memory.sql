-- Milestone 1: memory vertical slice.
--
-- Deviation from PRD Section 10.1's suggested table list: tags are stored
-- as a JSON array on memory_versions rather than in separate memory_tags /
-- memory_tag_links tables. Tag filtering uses SQLite's json_each() table
-- function. This keeps the v1 schema simpler; if tag-scale filtering
-- performance ever requires it, a join table can be introduced without
-- changing the domain model (MemoryVersion.tags stays string[]).

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  scope_level TEXT NOT NULL,
  scope_id TEXT,
  current_version_id TEXT NOT NULL,
  confidence REAL,
  importance REAL,
  valid_from TEXT,
  valid_until TEXT,
  created_at TEXT NOT NULL,
  created_by_type TEXT NOT NULL,
  created_by_id TEXT NOT NULL,
  created_by_display_name TEXT,
  updated_at TEXT NOT NULL,
  superseded_by_id TEXT,
  merged_into_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (status);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories (scope_level, scope_id);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories (kind);

CREATE TABLE IF NOT EXISTS memory_versions (
  id TEXT PRIMARY KEY,
  memory_id TEXT NOT NULL REFERENCES memories (id),
  version INTEGER NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  summary TEXT,
  tags TEXT NOT NULL DEFAULT '[]',
  structured_data TEXT,
  change_reason TEXT,
  created_at TEXT NOT NULL,
  created_by_type TEXT NOT NULL,
  created_by_id TEXT NOT NULL,
  created_by_display_name TEXT,
  UNIQUE (memory_id, version)
);

CREATE INDEX IF NOT EXISTS idx_memory_versions_memory_id ON memory_versions (memory_id);

CREATE TABLE IF NOT EXISTS provenance (
  id TEXT PRIMARY KEY,
  memory_id TEXT NOT NULL REFERENCES memories (id),
  memory_version_id TEXT REFERENCES memory_versions (id),
  source_type TEXT NOT NULL,
  source_ref TEXT NOT NULL,
  session_id TEXT,
  event_id TEXT,
  excerpt TEXT,
  artifact_hash TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_provenance_memory_id ON provenance (memory_id);

CREATE TABLE IF NOT EXISTS review_actions (
  id TEXT PRIMARY KEY,
  memory_id TEXT NOT NULL REFERENCES memories (id),
  memory_version_id TEXT NOT NULL REFERENCES memory_versions (id),
  action TEXT NOT NULL,
  reason TEXT,
  actor_type TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  actor_display_name TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_actions_memory_id ON review_actions (memory_id);

CREATE TABLE IF NOT EXISTS memory_relations (
  id TEXT PRIMARY KEY,
  from_memory_id TEXT NOT NULL REFERENCES memories (id),
  to_memory_id TEXT NOT NULL REFERENCES memories (id),
  type TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL,
  created_by_type TEXT NOT NULL,
  created_by_id TEXT NOT NULL,
  created_by_display_name TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_relations_from ON memory_relations (from_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_relations_to ON memory_relations (to_memory_id);

-- FTS5 external-content index over memory_versions (ADR 0008: unicode61,
-- remove_diacritics 2). content_rowid references memory_versions' implicit
-- integer rowid (distinct from its TEXT `id` primary key).
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  title,
  content,
  summary,
  tags,
  content = 'memory_versions',
  content_rowid = 'rowid',
  tokenize = 'unicode61 remove_diacritics 2'
);

-- memory_versions are append-only (never updated or deleted in normal
-- operation), so only INSERT/DELETE triggers are needed to keep the index
-- in sync.
CREATE TRIGGER IF NOT EXISTS memory_versions_ai AFTER INSERT ON memory_versions BEGIN
  INSERT INTO memory_fts (rowid, title, content, summary, tags)
  VALUES (new.rowid, new.title, new.content, new.summary, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memory_versions_ad AFTER DELETE ON memory_versions BEGIN
  INSERT INTO memory_fts (memory_fts, rowid, title, content, summary, tags)
  VALUES ('delete', old.rowid, old.title, old.content, old.summary, old.tags);
END;
