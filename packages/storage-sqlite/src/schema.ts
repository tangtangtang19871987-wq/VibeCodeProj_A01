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

export interface Database {
  settings: SettingsRow;
  projects: ProjectRow;
  agents: AgentRow;
}
