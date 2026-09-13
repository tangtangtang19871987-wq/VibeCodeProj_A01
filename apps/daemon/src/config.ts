import envPaths from "env-paths";
import * as path from "node:path";

/**
 * Data directory resolution per ADR 0004: `LAMS_DATA_DIR` overrides
 * unconditionally, otherwise the OS-conventional app data directory.
 * Never defaults to the current working directory (Section 19.1) so the
 * daemon behaves the same regardless of where it is launched from.
 */
export type LamsConfig = {
  dataDir: string;
  dbPath: string;
  artifactsDir: string;
  logsDir: string;
  host: string;
  port: number;
};

export function loadConfig(env: NodeJS.ProcessEnv = process.env): LamsConfig {
  const dataDir = env.LAMS_DATA_DIR ?? envPaths("lams", { suffix: "" }).data;
  const host = env.LAMS_HOST ?? "127.0.0.1";
  const port = env.LAMS_PORT ? Number(env.LAMS_PORT) : 4317;

  if (host !== "127.0.0.1" && host !== "localhost" && env.LAMS_UNSAFE_REMOTE !== "1") {
    throw new Error(
      `Refusing to bind to "${host}": LAMS binds to loopback by default (Section 18.1). ` +
        `Set LAMS_UNSAFE_REMOTE=1 to override at your own risk.`,
    );
  }

  return {
    dataDir,
    dbPath: path.join(dataDir, "db.sqlite3"),
    artifactsDir: path.join(dataDir, "artifacts"),
    logsDir: path.join(dataDir, "logs"),
    host,
    port,
  };
}
