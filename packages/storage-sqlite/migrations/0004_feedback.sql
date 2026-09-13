-- Milestone 3: memory_feedback, written by MCP's memory_feedback tool
-- (Section 12.2). Deferred from Milestone 2's migration because nothing
-- wrote to it until this tool exists.

CREATE TABLE IF NOT EXISTS memory_feedback (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions (id),
  recall_trace_id TEXT REFERENCES recall_traces (id),
  memory_id TEXT NOT NULL REFERENCES memories (id),
  event_type TEXT NOT NULL,
  evidence_level TEXT NOT NULL,
  note TEXT,
  outcome_metric TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_feedback_session_id ON memory_feedback (session_id);
CREATE INDEX IF NOT EXISTS idx_memory_feedback_memory_id ON memory_feedback (memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_feedback_trace_id ON memory_feedback (recall_trace_id);
