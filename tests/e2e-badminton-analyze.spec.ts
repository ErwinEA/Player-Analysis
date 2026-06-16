import path from "path";
import { test, expect } from "@playwright/test";

const TEST_VIDEO = path.resolve(
  __dirname,
  "../testing video/videoplayback.mp4",
);

test.describe("Full badminton analyze flow (videoplayback)", () => {
  test.describe.configure({ timeout: 90 * 60 * 1000 });

  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    const badge = page.locator('[class*="statusText"]');
    await expect(badge).toHaveText(/API (online|offline)/, {
      timeout: 25 * 60 * 1000,
    });
    test.skip(
      (await badge.innerText()) !== "API online",
      "Backend not online — start ./scripts/dev_backend.sh",
    );
  });

  test("upload videoplayback, near court + Jonah, show heat map and metrics", async ({
    page,
  }) => {
    await page.getByRole("radio", { name: "Badminton" }).click();

    await page.getByRole("radio", { name: "Near court" }).click();
    await page
      .getByRole("radiogroup", { name: "Primary shirt color" })
      .getByRole("radio", { name: "Blue" })
      .click();

    await page.getByLabel("Player name (optional)").fill("jonah");

    await page.locator("#upload-video-file-input").setInputFiles(TEST_VIDEO);

    await expect(page.locator("#video-filename")).toHaveText(
      /videoplayback\.mp4/i,
      { timeout: 60_000 },
    );
    await expect(
      page.getByRole("button", { name: "Change video" }),
    ).toBeVisible({ timeout: 60_000 });

    await expect(
      page.getByText(/Court corners saved for.*videoplayback/i),
    ).toBeVisible({ timeout: 30_000 });

    const analyze = page.getByRole("button", { name: "Analyze" });
    await expect(analyze).toBeEnabled({ timeout: 15_000 });

    await analyze.click();
    await expect(
      page.getByRole("button", { name: /^Analyzing/i }),
    ).toBeVisible({ timeout: 15_000 });

    // 12000-frame cap on MPS can take a long time.
    await expect(
      page.getByText("Player position heat map", { exact: true }),
    ).toBeVisible({ timeout: 80 * 60 * 1000 });
    await expect(page.getByText(/Analysis failed:/)).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: /^Analyze$/i }),
    ).toBeEnabled({ timeout: 5 * 60 * 1000 });

    await page.screenshot({
      path: "test-results/e2e-badminton-analyze-complete.png",
      fullPage: true,
    });

    const metrics = page.getByRole("region", { name: "Player Metrics" });
    await expect(metrics.getByText("Total rallies", { exact: true })).toBeVisible();
    await expect(
      metrics.getByText("Heuristic estimates — not official scoring.", { exact: true }),
    ).toBeVisible();

    const maskToggle = page.getByRole("checkbox", {
      name: /Show player mask/i,
    });
    if (await maskToggle.isVisible()) {
      await expect(maskToggle).toBeChecked();
    }
  });
});
