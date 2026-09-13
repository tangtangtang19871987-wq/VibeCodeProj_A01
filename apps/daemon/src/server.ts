import { buildApp } from "./app.js";
import { loadConfig } from "./config.js";
import { createContext } from "./context.js";

async function main() {
  const config = loadConfig();
  const ctx = createContext(config);
  const app = await buildApp(ctx);

  const closeGracefully = async (signal: string) => {
    app.log.info(`Received ${signal}, shutting down`);
    await app.close();
    ctx.close();
    process.exit(0);
  };
  process.on("SIGINT", () => void closeGracefully("SIGINT"));
  process.on("SIGTERM", () => void closeGracefully("SIGTERM"));

  await app.listen({ host: config.host, port: config.port });
  app.log.info(
    `LAMS daemon listening on http://${config.host}:${config.port} (data dir: ${config.dataDir})`,
  );
}

main().catch((err) => {
  console.error("Fatal error starting LAMS daemon:", err);
  process.exit(1);
});
