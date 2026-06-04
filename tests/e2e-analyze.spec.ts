import path from "path";
import { test, expect } from "@playwright/test";

const TEST_VIDEO = path.resolve(
  __dirname,
  "../testing video/testmatch2.mp4",
);

test.describe("Full analyze flow (testmatch2)", () => {
  test.describe.configure({ timeout: 60 * 60 * 1000 });

  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    const badge = page.locator('[class*="statusText"]');
    await expect(badge).toHaveText(/API (online|offline)/, { timeout: 25 * 60 * 1000 });
    test.skip(
      (await badge.innerText()) !== "API online",
      "Backend not online — start uvicorn on port 8000",
    );
  });

  test("upload testmatch2, analyze jersey 23, show heat map and metrics", async ({
    page,
  }) => {
    await page.getByLabel("Jersey number").fill("23");
    await page.getByLabel("Player name (optional)").fill("H.LOZANO");
    await page.getByRole("button", { name: "Green" }).first().click();
    await page
      .getByRole("group", { name: "Secondary / trim color" })
      .getByRole("button", { name: "White" })
      .click();

    await page.locator("#upload-video-file-input").setInputFiles(TEST_VIDEO);

    await expect(page.locator("#video-filename")).toHaveText(/testmatch2\.mp4/i, {
      timeout: 60_000,
    });
    await expect(
      page.getByRole("button", { name: "Change video" }),
    ).toBeVisible({ timeout: 60_000 });

    await expect(
      page.getByText(/Pitch corners saved for.*testmatch2/i),
    ).toBeVisible({ timeout: 30_000 });

    const analyze = page.getByRole("button", { name: "Analyze" });
    await expect(analyze).toBeEnabled({ timeout: 15_000 });

    await analyze.click();
    await expect(
      page.getByRole("button", { name: /^Analyzing/i }),
    ).toBeVisible({ timeout: 15_000 });

    await expect(
      page.getByRole("button", { name: /^Analyze$/i }),
    ).toBeVisible({ timeout: 45 * 60 * 1000 });
    await expect(
      page.getByRole("button", { name: /^Analyze$/i }),
    ).toBeEnabled();
    await expect(page.getByText(/Analysis failed:/)).toHaveCount(0);

    await expect(
      page.getByText("Player position heat map", { exact: true }),
    ).toBeVisible({ timeout: 60_000 });

    await page.screenshot({
      path: "test-results/e2e-analyze-complete.png",
      fullPage: true,
    });

    const metrics = page.getByRole("region", { name: "Player Metrics" });
    await expect(metrics.getByText(/km total/i)).toBeVisible();
    await expect(metrics.locator("definition").first()).not.toHaveText(/^—$/);

    const maskToggle = page.getByRole("checkbox", {
      name: /Show player mask/i,
    });
    if (await maskToggle.isVisible()) {
      await expect(maskToggle).toBeChecked();
    }
  });
});
