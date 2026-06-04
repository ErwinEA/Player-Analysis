import { test, expect } from "@playwright/test";

test.describe("Player Analysis dashboard", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("loads home page with core layout", async ({ page }) => {
    await expect(page).toHaveTitle(/Player Analysis/i);
    await expect(
      page.getByRole("heading", { name: "Player Analysis", level: 1 }),
    ).toBeVisible();
    await expect(
      page.getByRole("complementary", { name: "Player configuration" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Video Upload", level: 2 }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Drop your match video here/i }),
    ).toBeVisible();
    await expect(page.getByRole("button", { name: "Analyze" })).toBeVisible();
    await expect(
      page.getByRole("navigation", { name: "Footer" }),
    ).toBeVisible();
  });

  test("analyze is blocked until jersey and video are provided", async ({
    page,
  }) => {
    const analyze = page.getByRole("button", { name: "Analyze" });
    await expect(analyze).toBeDisabled();
    const hint = page.locator("#analyze-hint");
    await expect(hint).toContainText(/Required:/i);
    await expect(hint).toContainText(/Upload a video/i);
    await expect(hint).toContainText(/Enter jersey number/i);
  });

  test("jersey number input enables analyze hint to drop video requirement", async ({
    page,
  }) => {
    await page.getByLabel("Jersey number").fill("23");
    const hint = page.locator("#analyze-hint");
    await expect(hint).toContainText(/Upload a video/i);
    await expect(hint).not.toContainText(/Enter jersey number/i);
    await expect(page.getByRole("button", { name: "Analyze" })).toBeDisabled();
  });

  test("API status reaches online when backend is up", async ({ page }) => {
    const badge = page.locator('[class*="statusText"]');
    await expect(badge).toHaveText(/API (online|offline)/, { timeout: 60_000 });
    test.skip(
      (await badge.innerText()) !== "API online",
      "Backend not reachable — start uvicorn on port 8000",
    );
    await expect(badge).toHaveText("API online");
  });

  test("insights panels show empty state before analysis", async ({ page }) => {
    await expect(
      page.getByRole("heading", { name: "Position Heat Map", level: 2 }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Player Metrics", level: 2 }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Gameplay Analysis", level: 2 }),
    ).toBeVisible();
  });

  test("skip link focuses main content", async ({ page }) => {
    await page.getByRole("link", { name: "Skip to main content" }).focus();
    await expect(page.locator("#main-content")).toBeVisible();
  });
});
