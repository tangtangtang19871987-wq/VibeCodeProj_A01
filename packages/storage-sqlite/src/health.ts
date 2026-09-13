import type {
  ArtifactDirectoryHealthPort,
  DatabaseHealthPort,
  HealthCheckResult,
} from "@lams/application";
import * as fs from "node:fs";
import * as path from "node:path";
import { checkFts5Available, type SqliteConnection } from "./connection.js";
import { getAppliedMigrationIds, getPendingMigrationIds } from "./migrator.js";

export class SqliteHealthAdapter implements DatabaseHealthPort {
  constructor(private readonly conn: SqliteConnection) {}

  async checkConnection(): Promise<HealthCheckResult> {
    try {
      this.conn.raw.prepare("SELECT 1").get();
      return { ok: true };
    } catch (err) {
      return { ok: false, detail: (err as Error).message };
    }
  }

  async checkMigrations() {
    try {
      const appliedMigrations = getAppliedMigrationIds(this.conn.raw);
      const pendingMigrations = getPendingMigrationIds(this.conn.raw);
      return {
        ok: pendingMigrations.length === 0,
        detail:
          pendingMigrations.length > 0
            ? `${pendingMigrations.length} pending migration(s)`
            : undefined,
        appliedMigrations,
        pendingMigrations,
      };
    } catch (err) {
      return {
        ok: false,
        detail: (err as Error).message,
        appliedMigrations: [],
        pendingMigrations: [],
      };
    }
  }

  async checkFtsAvailable(): Promise<HealthCheckResult> {
    const ok = checkFts5Available(this.conn.raw);
    return ok ? { ok } : { ok, detail: "FTS5 module not available in this SQLite build" };
  }
}

export class FsArtifactDirectoryHealthAdapter
  implements ArtifactDirectoryHealthPort
{
  constructor(private readonly artifactDir: string) {}

  async checkWritable(): Promise<HealthCheckResult> {
    try {
      fs.mkdirSync(this.artifactDir, { recursive: true });
      const probe = path.join(this.artifactDir, ".write-probe");
      fs.writeFileSync(probe, "ok");
      fs.unlinkSync(probe);
      return { ok: true };
    } catch (err) {
      return { ok: false, detail: (err as Error).message };
    }
  }
}
