#!/usr/bin/env node
/**
 * One-command dev entry point (Milestone 0 acceptance criteria, ADR 0007):
 * runs the daemon and the Studio UI's Vite dev server together, so a
 * developer visits exactly one URL (the Vite dev server, which proxies
 * /api and /mcp to the daemon).
 */
import { spawn } from "node:child_process";

const daemon = spawn(
  "npx",
  ["tsx", "watch", "apps/daemon/src/server.ts"],
  { stdio: "inherit", shell: false },
);
const studio = spawn("npx", ["vite", "--config", "apps/studio/vite.config.ts"], {
  stdio: "inherit",
  shell: false,
  cwd: "apps/studio",
});

function shutdown() {
  daemon.kill("SIGTERM");
  studio.kill("SIGTERM");
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

daemon.on("exit", (code) => {
  if (code !== 0 && code !== null) {
    console.error(`daemon exited with code ${code}`);
    shutdown();
  }
});
studio.on("exit", (code) => {
  if (code !== 0 && code !== null) {
    console.error(`studio dev server exited with code ${code}`);
    shutdown();
  }
});
