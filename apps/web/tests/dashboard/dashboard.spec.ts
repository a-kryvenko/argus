import { test, expect, type Page } from "@playwright/test";

const admin = {
  id: 1,
  username: "andrew",
  active: true,
  groups: ["admins"],
  permissions: [
    "users.manage",
    "api_stats.read",
    "project_monitoring.read",
  ],
};
const checkedAt = new Date().toISOString();
const project = {
  status: "ok",
  checked_at: checkedAt,
  stale: false,
  services: [
    "Website",
    "API",
    "Clio",
    "Prophet",
    "Redis",
    "PostgreSQL (API)",
  ].map((name) => ({ name, status: "ok", response_ms: 12 })),
  host: {
    status: "ok",
    cpu_percent: 12.5,
    memory_percent: 42,
    disk_percent: 30,
    disk_free_bytes: 1024 ** 3 * 70,
  },
  observations: {
    status: "ok",
    sources: {
      kp: {
        source_id: "kp",
        label: "Estimated Kp",
        status: "ok",
        latest_observation_at: checkedAt,
        last_response_at: checkedAt,
        last_success_at: checkedAt,
        last_attempt_at: checkedAt,
        consecutive_failures: 0,
      },
    },
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
const summary = {
  requests: 12840,
  errors_4xx: 24,
  errors_5xx: 3,
  average_ms: 42.6,
  p95_upper_ms: 100,
};
const stats = {
  summary,
  hours: Array.from({ length: 24 }, (_, i) => ({
    ...summary,
    hour: `2026-09-11T${String(i).padStart(2, "0")}:00:00Z`,
    requests: 250 + ((i * 71) % 550),
    errors_4xx: i % 6,
    errors_5xx: i % 9 === 0 ? 1 : 0,
  })),
  routes: [
    "/public/observations/latest",
    "/public/solar-wind/history",
    "/public/geomagnetic/history",
  ].map((route, i) => ({
    ...summary,
    route,
    method: "GET",
    requests: 12840 - i * 2000,
  })),
  statuses: [
    { status: 200, requests: 12813 },
    { status: 404, requests: 24 },
    { status: 500, requests: 3 },
  ],
};
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
    else if (path === "/api-stats") data = stats;
    else if (path === "/project-monitoring") data = project;
    else if (path === "/project-traffic")
      data = {
        status: "ok",
        checked_at: checkedAt,
        stale: false,
        since: new Date(Date.now() - 86400000).toISOString(),
        resolution: "hour",
        recent_errors_5xx: 0,
        channels: Object.fromEntries(
          ["api", "site"].map((channel) => [
            channel,
            {
              summary: {
                ...summary,
                requests: channel === "api" ? 12840 : 3210,
              },
              points: [{ ...summary, time: checkedAt }],
            },
          ]),
        ),
      };
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
  await expect(page.getByText("12,840", { exact: true })).toBeVisible();
  await expect(page.locator("canvas").first()).toBeVisible();
  await expect(page.locator("h1")).toHaveCSS("font-size", "24px");
  await expect(page.getByText("Requests", { exact: true }).first()).toHaveCSS(
    "font-size",
    "12px",
  );
  await page.screenshot({
    path: "/tmp/argus-dashboard-overview.png",
    fullPage: true,
  });
  await expect(page.getByRole('navigation', { name: 'Analytics', exact: true }).getByRole('link', { name: 'Overview', exact: true })).toHaveAttribute('aria-current', 'page');
  await page.setViewportSize({ width: 1100, height: 1000 });
  await expect(page.getByRole('navigation', { name: 'Analytics', exact: true }).getByRole('link', { name: 'API statistics', exact: true })).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(
    page.getByRole("link", { name: "API statistics", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "API statistics", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "API statistics", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Endpoint performance")).toBeVisible();
  await page.getByLabel("Statistics period").selectOption("168");
  await expect(page.getByText("Across all API routes · 7 days")).toBeVisible();
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
  await page.screenshot({
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

test("mobile workspace menu closes on navigation; statistics remain contained", async ({
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
  await menu.getByRole('link', { name: 'API statistics', exact: true }).click();
  await expect(menu).toBeHidden();
  await expect(
    page.getByRole("heading", { name: "API statistics", exact: true }),
  ).toBeAttached();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
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
  await page.screenshot({
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
  page,
}) => {
  await mockApi(page);
  await page.route('**/api/v1/public/forecasts/**', route => route.fulfill({ status: 503, json: { success: false, data: null, error: { message: 'No test forecast' } } }));
  await page.goto("/");
  const before = await page
    .locator("h1")
    .first()
    .evaluate((element) => {
      const css = getComputedStyle(element);
      return { fontSize: css.fontSize, color: css.color, margin: css.margin };
    });
  await page.goto("/dashboard");
  await page.getByRole("navigation", { name: "Main navigation" }).getByRole("link", { name: "Forecast overview", exact: true }).click();
  await expect(page).toHaveURL("http://localhost:3000/");
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
  await expect(page.getByRole('heading', { name: 'LEO drag assessment', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Users & access', exact: true })).toHaveCount(0);
  await page.screenshot({ path: '/tmp/argus-member-workspace.png', fullPage: true });
  expect(
    reads.some((url) => /project-(monitoring|traffic)/.test(url.pathname)),
  ).toBeFalsy();
  await expect(page.getByText("Clio · Observation sources")).toHaveCount(0);
});

test("multiple groups use monitoring permission and show source and forecast timestamps", async ({
  page,
}) => {
  await mockApi(page, { ...admin, groups: ["clients", "admins"] });
  await page.goto("/dashboard");
  await expect(page.getByText("All systems operational")).toBeVisible();
  await expect(page.getByText("Estimated Kp", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Response received" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Published", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Public page requests", { exact: true }),
  ).toBeVisible();
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
  await expect(page.getByText("No current project status")).toBeVisible();
  await expect(page.getByText("All systems operational")).toHaveCount(0);
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
    page.getByText(/Could not refresh project status/),
  ).toBeVisible();
  await expect(
    page.getByText("Public page requests", { exact: true }),
  ).toBeVisible();
});

test("unconfigured traffic does not degrade healthy services", async ({ page }) => {
  await mockApi(page);
  await page.route("**/api/v1/dashboard/project-traffic?*", (route) =>
    route.fulfill({
      json: {
        success: true,
        data: {
          status: "disabled",
          checked_at: new Date().toISOString(),
          stale: false,
          since: null,
          recent_errors_5xx: 0,
          resolution: "hour",
          channels: null,
        },
      },
    }),
  );
  await page.goto("/dashboard");
  await expect(page.getByText("All systems operational")).toBeVisible();
  await expect(page.getByText(/Traffic collection is not configured/)).toBeVisible();
  await expect(page.getByText(/Counts may be incomplete/)).toHaveCount(0);
  await expect(page.getByText(/Collection started:/)).toHaveCount(0);
});

test("recent server errors override healthy service probes", async ({
  page,
}) => {
  await mockApi(page);
  await page.route("**/api/v1/dashboard/project-traffic?*", (route) =>
    route.fulfill({
      json: {
        success: true,
        data: {
          status: "ok",
          checked_at: checkedAt,
          stale: false,
          since: checkedAt,
          recent_errors_5xx: 7,
          resolution: "hour",
          channels: null,
        },
      },
    }),
  );
  await page.goto("/dashboard");
  await expect(page.getByText("Project needs attention")).toBeVisible();
  await expect(page.getByText(/7 server errors/)).toBeVisible();
  await expect(page.getByText("All systems operational")).toHaveCount(0);
});

for (const count of [0, "0"]) {
  test(`zero server errors (${typeof count}) do not degrade project health`, async ({
    page,
  }) => {
    await mockApi(page);
    await page.route("**/api/v1/dashboard/project-traffic?*", (route) =>
      route.fulfill({
        json: {
          success: true,
          data: {
            status: "ok",
            checked_at: new Date().toISOString(),
            stale: false,
            since: checkedAt,
            recent_errors_5xx: count,
            resolution: "hour",
            channels: null,
          },
        },
      }),
    );
    await page.goto("/dashboard");
    await expect(page.getByText("All systems operational")).toBeVisible();
    await expect(page.getByText(/server errors \(5xx\)/)).toHaveCount(0);
    await expect(page.getByText("Project needs attention")).toHaveCount(0);
  });
}

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
  for (const path of ['/dashboard/users', '/dashboard/api-stats']) {
    await page.goto(path);
    await expect(page.locator('.dashboard').getByRole('alert')).toContainText('You do not have access');
  }
  expect(reads.some(url => /\/(users|groups|api-stats|project-monitoring)/.test(url.pathname))).toBe(false);
});
