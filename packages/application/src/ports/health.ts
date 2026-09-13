/**
 * Health check ports. The daemon's /api/v1/health route (Section 19.3)
 * composes these into one report without knowing how each check works —
 * only `storage-sqlite` knows about SQLite/FTS5, only the daemon knows
 * about its own artifact directory path.
 */
export type HealthCheckResult = {
  ok: boolean;
  detail?: string;
};

export interface DatabaseHealthPort {
  checkConnection(): Promise<HealthCheckResult>;
  checkMigrations(): Promise<
    HealthCheckResult & { appliedMigrations: string[]; pendingMigrations: string[] }
  >;
  checkFtsAvailable(): Promise<HealthCheckResult>;
}

export interface ArtifactDirectoryHealthPort {
  checkWritable(): Promise<HealthCheckResult>;
}
