import { HealthService, ProjectService } from "@lams/application";
import {
  FsArtifactDirectoryHealthAdapter,
  openDatabase,
  runMigrations,
  SqliteHealthAdapter,
  SqliteProjectRepository,
  type SqliteConnection,
} from "@lams/storage-sqlite";
import type { LamsConfig } from "./config.js";

/**
 * Composition root: wires SQLite-backed repositories into application
 * services. This is the one place allowed to know about both `storage-sqlite`
 * and `@lams/application` concretely (Section 8.2, Milestone 0 acceptance
 * criteria on architecture boundaries).
 */
export type AppContext = {
  config: LamsConfig;
  conn: SqliteConnection;
  healthService: HealthService;
  projectService: ProjectService;
  close(): void;
};

export function createContext(config: LamsConfig): AppContext {
  const conn = openDatabase(config.dbPath);
  runMigrations(conn.raw);

  const dbHealth = new SqliteHealthAdapter(conn);
  const artifactHealth = new FsArtifactDirectoryHealthAdapter(config.artifactsDir);
  const healthService = new HealthService(dbHealth, artifactHealth);

  const projectRepo = new SqliteProjectRepository(conn.db);
  const projectService = new ProjectService(projectRepo);

  return {
    config,
    conn,
    healthService,
    projectService,
    close: () => conn.close(),
  };
}
