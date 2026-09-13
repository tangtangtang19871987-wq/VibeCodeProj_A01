-- Milestone 2: sessions and retrieval observability.
--
-- memory_feedback (the `injected`/`used`/`outcome_link` events reported via
-- MCP's memory_feedback tool) is deferred to the Milestone 3 migration,
-- since nothing writes to it until that tool exists. delivery_events is
-- created now because the recall pipeline itself writes `selected`/
-- `returned` rows (always daemon-observed, per ADR 0006) independent of MCP.

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  external_session_id TEXT,
  harness TEXT NOT NULL,
  agent_name TEXT,
  project_id TEXT NOT NULL REFERENCES projects (id),
  title TEXT,
  task_summary TEXT,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  outcome_result TEXT,
  outcome_summary TEXT,
  outcome_metric TEXT,
  metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_project_id ON sessions (project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status);

CREATE TABLE IF NOT EXISTS session_events (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions (id),
  type TEXT NOT NULL,
  payload_schema_version INTEGER NOT NULL,
  payload TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_events_session_id ON session_events (session_id);

CREATE TABLE IF NOT EXISTS recall_traces (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions (id),
  query TEXT NOT NULL,
  intent TEXT,
  scope_filter TEXT NOT NULL,
  kind_filter TEXT,
  status_filter TEXT NOT NULL,
  strategy TEXT NOT NULL,
  strategy_version TEXT NOT NULL,
  max_items INTEGER NOT NULL,
  budget_max_items INTEGER NOT NULL,
  budget_max_chars INTEGER NOT NULL,
  candidate_count INTEGER NOT NULL,
  selected_count INTEGER NOT NULL,
  returned_count INTEGER NOT NULL,
  returned_chars INTEGER NOT NULL,
  estimated_tokens INTEGER,
  latency_ms REAL NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recall_traces_session_id ON recall_traces (session_id);

CREATE TABLE IF NOT EXISTS recall_candidates (
  id TEXT PRIMARY KEY,
  recall_trace_id TEXT NOT NULL REFERENCES recall_traces (id),
  memory_id TEXT NOT NULL REFERENCES memories (id),
  memory_version_id TEXT NOT NULL REFERENCES memory_versions (id),
  eligible INTEGER NOT NULL,
  raw_score REAL NOT NULL,
  score_components TEXT NOT NULL,
  final_score REAL NOT NULL,
  rank INTEGER NOT NULL,
  selected INTEGER NOT NULL,
  returned INTEGER NOT NULL,
  exclusion_reasons TEXT NOT NULL,
  status_at_query_time TEXT NOT NULL,
  reason_codes TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recall_candidates_trace_id ON recall_candidates (recall_trace_id);
CREATE INDEX IF NOT EXISTS idx_recall_candidates_memory_id ON recall_candidates (memory_id);

CREATE TABLE IF NOT EXISTS delivery_events (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions (id),
  recall_trace_id TEXT REFERENCES recall_traces (id),
  memory_id TEXT NOT NULL REFERENCES memories (id),
  memory_version_id TEXT REFERENCES memory_versions (id),
  stage TEXT NOT NULL,
  evidence_level TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_delivery_events_session_id ON delivery_events (session_id);
CREATE INDEX IF NOT EXISTS idx_delivery_events_memory_id ON delivery_events (memory_id);
CREATE INDEX IF NOT EXISTS idx_delivery_events_trace_id ON delivery_events (recall_trace_id);
