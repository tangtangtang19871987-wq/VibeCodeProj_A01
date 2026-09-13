import { expect, test } from "@playwright/test";

/**
 * PRD Section 22.4 acceptance scenarios (subset covered here):
 * 1. create a draft memory
 * 2. review and approve it
 * 3. edit it and view the diff (version history)
 * 5. run a recall (Milestone 2) — not covered yet
 * 7. navigate memory -> detail and back
 * 8. confirm evidence labels — not covered yet (Milestone 2)
 *
 * Runs against a daemon started separately (see `pnpm test:e2e`), which
 * serves the prebuilt Studio UI on one port — no separate Vite dev server
 * needed for this test.
 */
test("create, approve, and edit a memory through the UI", async ({ page }) => {
  const uniqueTitle = `E2E test memory ${Date.now()}`;

  await page.goto("/memories");
  await expect(page.getByRole("heading", { name: "Memory Explorer" })).toBeVisible();

  // Create a draft memory.
  await page.getByPlaceholder("scope id (e.g. ilt-agent)").fill("e2e-project");
  await page.getByPlaceholder("title").fill(uniqueTitle);
  await page.getByPlaceholder("content").fill("Created by the Playwright smoke test.");
  await page.getByRole("button", { name: "Create draft memory" }).click();

  const row = page.getByRole("link", { name: uniqueTitle });
  await expect(row).toBeVisible();
  await expect(row.locator("xpath=ancestor::tr")).toContainText("draft");

  // Navigate to detail and approve it.
  await row.click();
  await expect(page.getByRole("heading", { name: uniqueTitle })).toBeVisible();
  await page.getByRole("button", { name: "Approve" }).click();
  await expect(page.locator(".status-badge")).toHaveText("verified");

  // Edit it — should create a new version, visible in version history.
  await page.getByRole("button", { name: "Edit (new version)" }).click();
  await page.locator(".stacked-form textarea").fill("Updated content from the smoke test.");
  await page.getByRole("button", { name: "Save as new version" }).click();

  await expect(page.locator(".memory-content")).toHaveText("Updated content from the smoke test.");
  await expect(page.getByText("v1", { exact: false }).first()).toBeVisible();
  await expect(page.getByText("v2", { exact: false }).first()).toBeVisible();

  // Navigate back to the Explorer and confirm the search finds it.
  await page.goto("/memories");
  await page.getByPlaceholder("Full-text search…").fill("smoke");
  await expect(page.getByRole("link", { name: uniqueTitle })).toBeVisible();
});
