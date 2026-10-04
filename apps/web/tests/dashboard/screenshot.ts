import type { Page } from "@playwright/test";

// These are manual review artifacts, not visual assertions. Failure screenshots
// remain enabled by Playwright independently of this opt-in.
export async function captureScreenshot(
  page: Page,
  options: NonNullable<Parameters<Page["screenshot"]>[0]>,
) {
  if (process.env.PLAYWRIGHT_SCREENSHOTS !== "1") return;
  await page.screenshot({
    animations: "disabled",
    style: "nextjs-portal { display: none; }",
    ...options,
  });
}
