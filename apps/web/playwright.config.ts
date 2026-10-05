import { defineConfig } from "@playwright/test";
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { demoFixture } from './tests/fixtures/demo';
const demoPath = path.join(mkdtempSync(path.join(tmpdir(), 'argus-demo-test-')), 'current.json');
writeFileSync(demoPath, JSON.stringify(demoFixture()));
export default defineConfig({
  testDir: "./tests/dashboard",
  fullyParallel: true,
  workers: 8,
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
    command: "pnpm exec next dev --turbopack --port 3100",
    url: "http://localhost:3100",
    reuseExistingServer: false,
    env: { NEXT_DIST_DIR: ".next/playwright", DEMO_BUNDLE_PATH: demoPath },
    timeout: 120000,
  },
});
