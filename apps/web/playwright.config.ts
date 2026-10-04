import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/dashboard",
  fullyParallel: true,
  workers: 2,
  timeout: 30000,
  expect: { timeout: 10000 },
  use: {
    baseURL: "http://localhost:3100",
    viewport: { width: 1440, height: 1000 },
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {},
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "pnpm exec next dev --webpack --port 3100",
    url: "http://localhost:3100",
    reuseExistingServer: false,
    env: { NEXT_DIST_DIR: ".next/playwright" },
    timeout: 120000,
  },
});
