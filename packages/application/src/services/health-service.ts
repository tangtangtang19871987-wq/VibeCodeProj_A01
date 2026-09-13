import type {
  ArtifactDirectoryHealthPort,
  DatabaseHealthPort,
} from "../ports/health.js";

export type HealthReport = {
  ok: boolean;
  checks: {
    process: { ok: true };
    database: { ok: boolean; detail?: string };
    migrations: {
      ok: boolean;
      detail?: string;
      appliedMigrations: string[];
      pendingMigrations: string[];
    };
    fts: { ok: boolean; detail?: string };
    artifactDirectory: { ok: boolean; detail?: string };
  };
};

/**
 * PRD Section 19.3: distinguish process-alive, db-reachable/migrated,
 * FTS-available, and artifact-dir-writable rather than one boolean.
 */
export class HealthService {
  constructor(
    private readonly db: DatabaseHealthPort,
    private readonly artifacts: ArtifactDirectoryHealthPort,
  ) {}

  async check(): Promise<HealthReport> {
    const [database, migrations, fts, artifactDirectory] = await Promise.all([
      this.db.checkConnection(),
      this.db.checkMigrations(),
      this.db.checkFtsAvailable(),
      this.artifacts.checkWritable(),
    ]);

    const ok = database.ok && migrations.ok && fts.ok && artifactDirectory.ok;

    return {
      ok,
      checks: {
        process: { ok: true },
        database,
        migrations,
        fts,
        artifactDirectory,
      },
    };
  }
}
