import { existsSync } from "node:fs";
import { defineConfig } from "@playwright/test";

// This repo's sandboxed dev environment pre-installs Chromium outside
// Playwright's normal cache and points PLAYWRIGHT_BROWSERS_PATH at it; use
// that binary directly when present, otherwise fall back to Playwright's
// own managed browser (requires `npx playwright install chromium`, as CI
// does) so this config also works on a plain checkout/CI runner.
const sandboxChromium = "/opt/pw-browsers/chromium";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:4318",
    trace: "retain-on-failure",
    launchOptions: existsSync(sandboxChromium)
      ? { executablePath: sandboxChromium }
      : {},
  },
});
