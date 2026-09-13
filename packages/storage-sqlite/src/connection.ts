import Database from "better-sqlite3";
import { Kysely, SqliteDialect } from "kysely";
import * as fs from "node:fs";
import * as path from "node:path";
import type { Database as Schema } from "./schema.js";

export type LamsDatabase = Kysely<Schema>;

export type SqliteConnection = {
  raw: Database.Database;
  db: LamsDatabase;
  close(): void;
};

/**
 * Opens the SQLite database in WAL mode with foreign keys enabled
 * (PRD Section 10.1). `filePath` may be `:memory:` for tests.
 */
export function openDatabase(filePath: string): SqliteConnection {
  if (filePath !== ":memory:") {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
  }

  const raw = new Database(filePath);
  raw.pragma("journal_mode = WAL");
  raw.pragma("foreign_keys = ON");
  raw.pragma("busy_timeout = 5000");

  const db = new Kysely<Schema>({
    dialect: new SqliteDialect({ database: raw }),
  });

  return {
    raw,
    db,
    close: () => {
      raw.close();
    },
  };
}

/** Whether the SQLite build linked in has FTS5 support (Section 10, 22.6). */
export function checkFts5Available(raw: Database.Database): boolean {
  try {
    raw.exec("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_probe USING fts5(x)");
    raw.exec("DROP TABLE IF EXISTS __fts5_probe");
    return true;
  } catch {
    return false;
  }
}
