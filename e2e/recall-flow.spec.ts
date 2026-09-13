import { expect, test } from "@playwright/test";

/**
 * PRD Section 22.4 scenarios: run a recall, inspect ranking reasons,
 * navigate recall -> memory -> source session and back, confirm evidence
 * labels.
 */
test("create a verified memory, open a session, run a recall, and inspect it", async ({ page }) => {
  const unique = Date.now();
  const projectKey = `e2e-recall-${unique}`;
  const memoryTitle = `Recall test memory ${unique}`;

  // Create a project via the Dashboard.
  await page.goto("/");
  await page.getByPlaceholder("key (e.g. ilt-agent)").fill(projectKey);
  await page.getByPlaceholder("display name").fill("E2E Recall Project");
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByText(projectKey, { exact: false })).toBeVisible();

  // Create and approve a memory in that project's scope.
  await page.goto("/memories");
  await page.getByLabel("scope project").selectOption({ label: `E2E Recall Project (${projectKey})` });
  await page.getByPlaceholder("title").fill(memoryTitle);
  await page
    .getByPlaceholder("content")
    .fill("Zephyrwidget calibration requires a 12-hour thermal soak before measurement.");
  await page.getByRole("button", { name: "Create draft memory" }).click();
  await page.getByRole("link", { name: memoryTitle }).click();
  await page.getByRole("button", { name: "Approve" }).click();
  await expect(page.locator(".status-badge")).toHaveText("verified");

  // Open a session for that project.
  await page.goto("/sessions");
  await page.getByLabel("project").selectOption({ label: `E2E Recall Project (${projectKey})` });
  await page.getByPlaceholder("task summary (optional)").fill("e2e recall task");
  await page.getByRole("button", { name: "Open session" }).click();
  await page.getByRole("link", { name: "e2e recall task" }).click();
  await expect(page.getByRole("heading", { name: "e2e recall task" })).toBeVisible();

  // Run a recall that should find the memory just created.
  await page.getByPlaceholder("query").fill("Zephyrwidget thermal soak");
  await page.getByRole("button", { name: "Recall" }).click();
  await expect(page.getByText(/Returned 1 memory/)).toBeVisible();

  // Follow the link into the Retrieval Inspector.
  await page.getByRole("link", { name: "inspect this recall" }).click();
  await expect(page.getByRole("heading", { name: "Retrieval Inspector" })).toBeVisible();
  await expect(page.getByText(memoryTitle)).toBeVisible();
  await expect(page.locator(".evidence-tag--observed").first()).toBeVisible();

  // Navigate from the candidate back to the memory.
  await page.getByRole("link", { name: memoryTitle }).click();
  await expect(page.getByRole("heading", { name: memoryTitle })).toBeVisible();
});
