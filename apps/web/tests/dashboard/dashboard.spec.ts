import { captureScreenshot } from "./screenshot";
import { test, expect, type Page } from "@playwright/test";

const admin = {
  id: 1,
  username: "andrew",
  active: true,
  groups: ["admins"],
  permissions: [
    "users.manage",
    "project_monitoring.read",
  ],
};
const checkedAt = new Date().toISOString();
const project = {
  status: "ok",
  checked_at: checkedAt,
  stale: false,
  observations: {
    status: "ok",
    measurements: [
      {
        metric: "dst",
        status: "fresh",
        latest_observation_at: checkedAt,
        received_at: checkedAt,
        stale_after_seconds: 21600,
      },
    ],
  },
  forecasts: [
    {
      product: "dst",
      status: "ok",
      freshness: "within_age_limit",
      current_release: { issue_time: checkedAt, published_at: checkedAt },
      latest_attempt: {
        status: "succeeded",
        started_at: checkedAt,
        finished_at: checkedAt,
      },
      latest_attempt_artifacts: [],
    },
  ],
};
const secondUser = { ...admin, id: 2, username: "alex" };
async function mockApi(page: Page, user = admin) {
  const writes: {
    method: string;
    path: string;
    body: Record<string, unknown>;
  }[] = [];
  const reads: URL[] = [];
  await page.route("**/api/v1/dashboard/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace("/api/v1/dashboard", "");
    if (request.method() !== "GET") {
      const body = request.postDataJSON() ?? {};
      writes.push({ method: request.method(), path, body });
      if (path === "/login" && body.password === "incorrect") {
        await route.fulfill({
          status: 401,
          json: {
            success: false,
            data: null,
            error: { message: "Invalid username or password" },
          },
        });
        return;
      }
      await route.fulfill({ json: { success: true, data: user, error: null } });
      return;
    }
    reads.push(url);
    let data: unknown;
    if (path === "/me") data = user;
    else if (path === "/project-monitoring") data = project;

    else if (path === "/users") data = { items: [admin, secondUser], total: 2 };
    else if (path === "/groups")
      data = [{ name: "admins", permissions: admin.permissions }];
    await route.fulfill({ json: { success: true, data, error: null } });
  });
  return { writes, reads };
}

test("overview, chart and shared workspace navigation", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await mockApi(page);
  await page.goto("/dashboard");
  await expect(
    page.getByRole("heading", { name: "Project overview" }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Clio · Observations" })).toBeVisible();
  await expect(page.getByRole("link", { name: "API statistics", exact: true })).toHaveCount(0);
  await expect(page.locator("h1")).toHaveCSS("font-size", "24px");
  await captureScreenshot(page, {
    path: "/tmp/argus-dashboard-overview.png",
    fullPage: true,
  });
  await expect(page.getByRole('navigation', { name: 'Analytics', exact: true }).getByRole('link', { name: 'Overview', exact: true })).toHaveAttribute('aria-current', 'page');
  expect(errors).toEqual([]);
});

test("create and edit dialogs preserve groups and submit correct mutations", async ({
  page,
}) => {
  const { writes } = await mockApi(page);
  await page.goto("/dashboard/users");
  await page.getByRole("button", { name: "Create user", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveCSS("position", "fixed");
  await expect(dialog).toHaveCSS("background-color", "rgb(16, 23, 30)");
  await dialog.getByLabel("Username", { exact: true }).fill("newuser");
  await dialog
    .getByLabel("Password", { exact: true })
    .fill("new-test-password");
  await dialog.getByRole("checkbox", { name: "admins" }).check();
  await dialog
    .getByRole("button", { name: "Create user", exact: true })
    .click();
  await expect(dialog).not.toBeVisible();
  expect(writes.find((write) => write.path === "/users")?.body).toEqual({
    username: "newuser",
    password: "new-test-password",
    groups: ["admins"],
  });
  await page.getByRole("button", { name: "Edit alex", exact: true }).click();
  await expect(dialog.getByRole("checkbox", { name: "admins" })).toBeChecked();
  await dialog.getByLabel("Account status").selectOption("false");
  await expect(dialog).toHaveCSS("opacity", "1");
  await captureScreenshot(page, {
    animations: "disabled",
    path: "/tmp/argus-dashboard-user-dialog.png",
    fullPage: true,
  });
  await dialog.getByRole("button", { name: "Save changes" }).click();
  await expect(dialog).not.toBeVisible();
  expect(writes.find((write) => write.path === "/users/2")?.body).toEqual({
    active: false,
    groups: ["admins"],
  });
});

test("mobile workspace menu closes on navigation; users remain contained", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockApi(page);
  await page.goto("/dashboard");
  await expect(
    page.getByRole("heading", { name: "Project overview" }),
  ).toBeVisible();
  await page.getByText('Resources', { exact: true }).click();
  const menu = page.getByRole('navigation', { name: 'Mobile resources' });
  await expect(menu).toBeVisible();
  await menu.getByRole('link', { name: 'Users & access', exact: true }).click();
  await expect(menu).toBeHidden();
  await expect(
    page.getByRole("heading", { name: "Users & access", exact: true }),
  ).toBeAttached();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await captureScreenshot(page, {
    path: "/tmp/argus-dashboard-mobile.png",
    fullPage: true,
  });
});

test("login errors, sign in and sign out", async ({ page }) => {
  const { writes } = await mockApi(page);
  await page.goto("/dashboard/login");
  await expect(page.getByText("Welcome back", { exact: true })).toBeVisible();
  await page.getByLabel("Username", { exact: true }).fill("andrew");
  await page.getByLabel("Password", { exact: true }).fill("incorrect");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.locator(".dashboard").getByRole("alert")).toContainText(
    "Invalid username or password",
  );
  await captureScreenshot(page, {
    path: "/tmp/argus-dashboard-login.png",
    fullPage: true,
  });
  await page
    .getByLabel("Password", { exact: true })
    .fill("correct-test-password");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Project overview" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Account menu" }).click();
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page).toHaveURL(/\/dashboard\/login$/);
  expect(writes.some((write) => write.path === "/logout")).toBeTruthy();
});

test("permissions hide protected navigation and direct pages", async ({
  page,
}) => {
  await mockApi(page, { ...admin, groups: [], permissions: [] });
  await page.goto("/dashboard/users");
  await expect(page.locator(".dashboard").getByRole("alert")).toContainText(
    "You do not have access",
  );
  await expect(
    page
      .getByRole("complementary", { name: "Workspace navigation" })
      .getByRole("link", { name: "Users & access", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Create user", exact: true }),
  ).toHaveCount(0);
});

test("dashboard styles do not change public typography after client navigation", async ({
  page, baseURL,
}) => {
  await mockApi(page);
  await page.route('**/api/v1/forecasts/**', route => route.fulfill({ status: 503, json: { success: false, data: null, error: { message: 'No test forecast' } } }));
  await page.goto("/");
  const before = await page
    .locator("h1")
    .first()
    .evaluate((element) => {
      const css = getComputedStyle(element);
      return { fontSize: css.fontSize, color: css.color, margin: css.margin };
    });
  await page.goto("/dashboard");
  await page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: "Overview", exact: true }).click();
  await expect(page).toHaveURL(`${baseURL}/`);
  await expect(page.locator(".dashboard")).toHaveCount(0);
  const after = await page
    .locator("h1")
    .first()
    .evaluate((element) => {
      const css = getComputedStyle(element);
      return { fontSize: css.fontSize, color: css.color, margin: css.margin };
    });
  expect(after).toEqual(before);
});

test("client landing never requests project monitoring", async ({ page }) => {
  const { reads } = await mockApi(page, {
    ...admin,
    groups: ["clients"],
    permissions: [],
  });
  await page.goto("/dashboard");
  await expect(
    page.getByRole("heading", { name: "Your workspace" }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Drag risk', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Users & access', exact: true })).toHaveCount(0);
  await captureScreenshot(page, { path: '/tmp/argus-member-workspace.png', fullPage: true });
  expect(
    reads.some((url) => /project-(monitoring|traffic)/.test(url.pathname)),
  ).toBeFalsy();
  await expect(page.getByText("Clio · Observations")).toHaveCount(0);
});

test("multiple groups use monitoring permission and show observation and forecast timestamps", async ({
  page,
}) => {
  await mockApi(page, { ...admin, groups: ["clients", "admins"] });
  await page.goto("/dashboard");
  await expect(page.getByText("Observations and forecasts are current")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Clio · Observations", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "Services", exact: true })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Server resources" })).toHaveCount(0);
  await expect(
    page.getByRole("columnheader", { name: "Last received" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Published", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Project overview", exact: true })).toBeVisible();
});

test("stale checks cannot display a healthy project", async ({ page }) => {
  await mockApi(page);
  await page.route("**/api/v1/dashboard/project-monitoring", (route) =>
    route.fulfill({
      json: {
        success: true,
        data: { ...project, checked_at: "2020-01-01T00:00:00Z" },
      },
    }),
  );
  await page.goto("/dashboard");
  await expect(page.getByText("No current data status")).toBeVisible();
  await expect(page.getByText("Observations and forecasts are current")).toHaveCount(0);
});

test("unavailable monitoring is visible and does not break the page", async ({
  page,
}) => {
  await mockApi(page);
  await page.route("**/api/v1/dashboard/project-monitoring", (route) =>
    route.fulfill({
      status: 503,
      json: { success: false, error: { message: "Unavailable" } },
    }),
  );
  await page.goto("/dashboard");
  await expect(
    page.getByText(/Could not refresh data status/),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Project overview", exact: true })).toBeVisible();
});

test('mobile member can open account and sign out from shared resources', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const { writes } = await mockApi(page, { ...admin, groups: ['clients'], permissions: [] });
  await page.goto('/dashboard');
  await expect(page.getByRole('heading', { name: 'Your workspace' })).toBeVisible();
  await page.getByText('Resources', { exact: true }).click();
  const menu = page.getByRole('navigation', { name: 'Mobile resources' });
  await expect(menu.getByRole('link', { name: 'Users & access' })).toHaveCount(0);
  await menu.getByRole('button', { name: 'Account menu' }).click();
  await menu.getByRole('button', { name: 'Sign out', exact: true }).click();
  await expect(page).toHaveURL(/\/dashboard\/login$/);
  expect(writes.some(write => write.path === '/logout')).toBe(true);
});

test('restricted sections do not fetch protected data for a member', async ({ page }) => {
  const { reads } = await mockApi(page, { ...admin, groups: ['clients'], permissions: [] });
  for (const path of ['/dashboard/users']) {
    await page.goto(path);
    await expect(page.locator('.dashboard').getByRole('alert')).toContainText('You do not have access');
  }
  expect(reads.some(url => /\/(users|groups|project-monitoring)/.test(url.pathname))).toBe(false);
});
