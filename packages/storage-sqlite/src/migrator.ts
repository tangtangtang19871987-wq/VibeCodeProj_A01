import type Database from "better-sqlite3";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const currentDir = path.dirname(fileURLToPath(import.meta.url));
// This module lives at <package-root>/src/migrator.ts (dev) or
// <package-root>/dist/migrator.js (build) — both one level below package
// root, so the migrations directory is always found via the same relative
// path regardless of which one is running.
export const MIGRATIONS_DIR = path.resolve(currentDir, "../migrations");

export type MigrationInfo = {
  id: string;
  fileName: string;
};

function bootstrapMigrationsTable(raw: Database.Database): void {
  raw.exec(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      id TEXT PRIMARY KEY,
      applied_at TEXT NOT NULL
    )
  `);
}

export function listMigrationFiles(dir = MIGRATIONS_DIR): MigrationInfo[] {
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".sql"))
    .sort()
    .map((fileName) => ({ id: fileName.replace(/\.sql$/, ""), fileName }));
}

export function getAppliedMigrationIds(raw: Database.Database): string[] {
  bootstrapMigrationsTable(raw);
  const rows = raw
    .prepare("SELECT id FROM schema_migrations ORDER BY id ASC")
    .all() as { id: string }[];
  return rows.map((r) => r.id);
}

/**
 * Applies every migration file not yet recorded in `schema_migrations`, in
 * filename order, each inside its own transaction. Never rewrites a
 * migration that has already been applied.
 */
export function runMigrations(
  raw: Database.Database,
  dir = MIGRATIONS_DIR,
): { applied: string[] } {
  bootstrapMigrationsTable(raw);
  const applied = new Set(getAppliedMigrationIds(raw));
  const files = listMigrationFiles(dir);
  const newlyApplied: string[] = [];

  for (const migration of files) {
    if (applied.has(migration.id)) continue;
    const sql = fs.readFileSync(path.join(dir, migration.fileName), "utf-8");
    const apply = raw.transaction(() => {
      raw.exec(sql);
      raw
        .prepare(
          "INSERT INTO schema_migrations (id, applied_at) VALUES (?, ?)",
        )
        .run(migration.id, new Date().toISOString());
    });
    apply();
    newlyApplied.push(migration.id);
  }

  return { applied: newlyApplied };
}

export function getPendingMigrationIds(
  raw: Database.Database,
  dir = MIGRATIONS_DIR,
): string[] {
  const applied = new Set(getAppliedMigrationIds(raw));
  return listMigrationFiles(dir)
    .map((m) => m.id)
    .filter((id) => !applied.has(id));
}
