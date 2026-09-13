#!/usr/bin/env node
/**
 * Runs the Playwright smoke tests (Section 22.4) against a real daemon
 * serving the prebuilt Studio UI, using a throwaway data directory so
 * tests never touch a developer's real LAMS data.
 */
import { spawn, spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "lams-e2e-"));
const port = 4318;

console.log(`[e2e] building...`);
const build = spawnSync("pnpm", ["build"], { stdio: "inherit" });
if (build.status !== 0) process.exit(build.status ?? 1);

console.log(`[e2e] starting daemon on port ${port} (data dir: ${dataDir})`);
const daemon = spawn("node", ["apps/daemon/dist/server.js"], {
  stdio: "inherit",
  env: { ...process.env, LAMS_DATA_DIR: dataDir, LAMS_PORT: String(port), NODE_ENV: "production" },
});

async function waitForHealth(): Promise<void> {
  for (let i = 0; i < 50; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/api/v1/health`);
      if (res.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error("daemon did not become healthy in time");
}

async function main() {
  try {
    await waitForHealth();
    console.log("[e2e] daemon healthy, running Playwright...");
    const result = spawnSync("npx", ["playwright", "test"], { stdio: "inherit" });
    process.exitCode = result.status ?? 1;
  } finally {
    daemon.kill("SIGTERM");
    fs.rmSync(dataDir, { recursive: true, force: true });
  }
}

main();
