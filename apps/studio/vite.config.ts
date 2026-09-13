import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const DAEMON_URL = process.env.LAMS_DAEMON_URL ?? "http://127.0.0.1:4317";

// ADR 0007: in dev, Vite proxies /api and /mcp to the daemon so the
// developer only ever visits one URL (the Vite dev server's).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: DAEMON_URL, changeOrigin: true },
      "/mcp": { target: DAEMON_URL, changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
