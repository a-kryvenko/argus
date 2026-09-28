import { test, expect } from "@playwright/test";

const inputs = { altitude_km: 400, inclination_deg: 51.6, mass_kg: 100, effective_area_m2: 1, drag_coefficient: 2.2, horizon_hours: 24 };
const issue = "2026-09-28T00:00:00Z";
const result = {
  inputs, start_time: issue, end_time: "2026-09-29T00:00:00Z", computed_at: issue,
  estimated_altitude_loss_m: 86.4, delta_v_loss_m_s: 0.048, mean_density_kg_m3: 1e-12, mean_drag_accel_m_s2: 5.6e-7,
  source: { issue_time: issue, observed_at: issue, dtc_observed_at: issue, density_model: "JB2008" },
  assumptions: ["Circular orbit; constant area and drag coefficient."],
  predictions: Array.from({ length: 25 }, (_, i) => ({
    lead_hours: i, valid_time: new Date(Date.parse(issue) + i * 3600000).toISOString(),
    estimated_altitude_loss_m: i * 3.6, delta_v_loss_m_s: i * 0.002, mean_density_kg_m3: 1e-12, mean_drag_accel_m_s2: 5.6e-7,
  })),
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/dashboard/me", route => route.fulfill({ json: {
    success: true, data: { id: 2, username: "member", active: true, groups: [], permissions: [] }, error: null,
  } }));
});

test("member can calculate, inspect results and retry after unavailable data", async ({ page }) => {
  let calls = 0;
  await page.route("**/api/v1/public/risks/leo-drag", async route => {
    calls++;
    expect(route.request().postDataJSON()).toEqual(inputs);
    await route.fulfill(calls === 1
      ? { json: { success: true, data: result, error: null } }
      : { status: 503, json: { success: false, data: null, error: { message: "Not ready" } } });
  });
  await page.goto("/dashboard/risk/leo");
  await expect(page.getByRole("heading", { name: "LEO drag assessment" })).toBeVisible();
  await expect(page.locator('a[href="/dashboard/risk/leo"]')).toHaveAttribute("aria-current", "page");
  expect(calls).toBe(0);
  await page.getByRole("button", { name: "Calculate drag" }).click();
  await expect(page.getByText("86.4 m", { exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: /Cumulative altitude loss/ }).locator("canvas")).toBeVisible();
  await page.getByText("Hourly estimates", { exact: true }).click();
  await expect(page.getByRole("row")).toHaveCount(26);
  await page.getByLabel("Mass (kg)").fill("200");
  // An edited form must not relabel the previous result as a new calculation.
  await expect(page.getByText(/Calculated for.*100 kg/)).toBeVisible();
  await page.getByLabel("Mass (kg)").fill("100");
  await page.screenshot({ path: "test-results/leo-drag-desktop.png", fullPage: true });
  await page.getByRole("button", { name: "Calculate drag" }).click();
  await expect(page.locator('.dashboard [role="alert"]')).toContainText("Drag assessment is unavailable");
  await expect(page.getByText("86.4 m", { exact: true })).toHaveCount(0);
  expect(calls).toBe(2);
});

test("validates positive area and supports a mobile 48-hour request", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  let calls = 0;
  await page.route("**/api/v1/public/risks/leo-drag", async route => {
    calls++;
    expect(route.request().postDataJSON()).toEqual({ ...inputs, horizon_hours: 48 });
    await route.fulfill({ status: 422, json: { success: false, data: null, error: { message: "Outside grid" } } });
  });
  await page.goto("/dashboard/risk/leo");
  await page.getByLabel("Effective area (m²)").fill("0");
  await page.getByRole("button", { name: "Calculate drag" }).click();
  await expect(page.locator('.dashboard [role="alert"]')).toContainText("greater than zero");
  expect(calls).toBe(0);
  await page.getByLabel("Effective area (m²)").fill("1");
  await page.getByLabel("Assessment horizon").selectOption("48");
  await page.getByRole("button", { name: "Calculate drag" }).click();
  await expect(page.locator('.dashboard [role="alert"]')).toContainText("model limits");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: "test-results/leo-drag-mobile.png", fullPage: true });
});

test("requires a dashboard session", async ({ page }) => {
  await page.route("**/api/v1/dashboard/me", route => route.fulfill({ status: 401,
    json: { success: false, data: null, error: { message: "Sign in" } } }));
  await page.goto("/dashboard/risk/leo");
  await expect(page).toHaveURL(/\/dashboard\/login/);
});
