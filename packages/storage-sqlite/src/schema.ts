/**
 * Kysely `Database` interface — the explicit, hand-maintained mapping onto
 * the SQL tables created by migrations (ADR 0002). Extended as each
 * milestone's migration adds tables; never auto-generated.
 */
export interface SettingsRow {
  key: string;
  value: string;
  updated_at: string;
}

export interface ProjectRow {
  id: string;
  key: string;
  name: string;
  description: string | null;
  created_at: string;
}

export interface AgentRow {
  id: string;
  name: string;
  harness: string;
  created_at: string;
}

export interface MemoryRow {
  id: string;
  kind: string;
  status: string;
  scope_level: string;
  scope_id: string | null;
  current_version_id: string;
  confidence: number | null;
  importance: number | null;
  valid_from: string | null;
  valid_until: string | null;
  created_at: string;
  created_by_type: string;
  created_by_id: string;
  created_by_display_name: string | null;
  updated_at: string;
  superseded_by_id: string | null;
  merged_into_id: string | null;
}

export interface MemoryVersionRow {
  id: string;
  memory_id: string;
  version: number;
  title: string;
  content: string;
  summary: string | null;
  tags: string;
  structured_data: string | null;
  change_reason: string | null;
  created_at: string;
  created_by_type: string;
  created_by_id: string;
  created_by_display_name: string | null;
}

export interface ProvenanceRow {
  id: string;
  memory_id: string;
  memory_version_id: string | null;
  source_type: string;
  source_ref: string;
  session_id: string | null;
  event_id: string | null;
  excerpt: string | null;
  artifact_hash: string | null;
  created_at: string;
}

export interface ReviewActionRow {
  id: string;
  memory_id: string;
  memory_version_id: string;
  action: string;
  reason: string | null;
  actor_type: string;
  actor_id: string;
  actor_display_name: string | null;
  created_at: string;
}

export interface MemoryRelationRow {
  id: string;
  from_memory_id: string;
  to_memory_id: string;
  type: string;
  note: string | null;
  created_at: string;
  created_by_type: string;
  created_by_id: string;
  created_by_display_name: string | null;
}

export interface MemoryFtsRow {
  rowid: number;
  title: string;
  content: string;
  summary: string | null;
  tags: string;
  rank: number;
}

export interface Database {
  settings: SettingsRow;
  projects: ProjectRow;
  agents: AgentRow;
  memories: MemoryRow;
  memory_versions: MemoryVersionRow;
  provenance: ProvenanceRow;
  review_actions: ReviewActionRow;
  memory_relations: MemoryRelationRow;
  memory_fts: MemoryFtsRow;
}
