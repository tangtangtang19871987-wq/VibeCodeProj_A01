#!/usr/bin/env node
import { runStdioShim } from "@lams/stdio-shim";
import { loadConfig } from "./config.js";
import { createContext } from "./context.js";

/**
 * PRD Section 19.2. `start`/`dev` run the daemon in the foreground for v1
 * (OS service installation is explicitly deferred, per Section 19.2's own
 * "do not make OS service integration a blocker" instruction). `status` and
 * `doctor` are read-only diagnostics that open their own short-lived
 * connection rather than talking to a running process, since v1 has no
 * inter-process control channel yet.
 */
async function main() {
  const [, , command] = process.argv;

  switch (command) {
    case "start":
    case "dev": {
      await import("./server.js");
      return;
    }
    case "status":
    case "doctor": {
      const config = loadConfig();
      const ctx = createContext(config);
      try {
        const report = await ctx.healthService.check();
        console.log(JSON.stringify(report, null, 2));
        process.exitCode = report.ok ? 0 : 1;
      } finally {
        ctx.close();
      }
      return;
    }
    case "mcp-stdio": {
      const config = loadConfig();
      const daemonUrl = `http://${config.host}:${config.port}/mcp`;
      await runStdioShim(daemonUrl);
      return;
    }
    case "stop":
    case "backup":
    case "export":
    case "import": {
      console.error(
        `"${command}" is not implemented yet (planned for a later milestone).`,
      );
      process.exitCode = 1;
      return;
    }
    default: {
      console.error(
        "Usage: lams <start|dev|status|doctor|mcp-stdio|stop|backup|export|import>",
      );
      process.exitCode = 1;
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
