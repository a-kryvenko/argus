import { test, expect, type Page } from "@playwright/test";

const admin = {
  id: 1,
  username: "andrew",
  active: true,
  groups: ["admins"],
  permissions: [
    "observations.read",
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
    "/dashboard/observations",
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
    else if (path === "/observations") {
      const normalized = url.searchParams.get("kind") === "normalized";
      data = {
        columns: normalized
          ? [
              "observed_at",
              "bx",
              "by",
              "bz",
              "v",
              "n",
              "t",
              "kp",
              "dst",
              "ap",
              "f10_7",
              "s10",
              "m10",
              "y10",
            ]
          : ["id", "metric", "value", "observed_at"],
        items: Array.from({ length: 8 }, (_, i) =>
          normalized
            ? {
                observed_at: `2026-09-11T${String(i).padStart(2, "0")}:00:00Z`,
                bx: 1.2,
                by: -2.5,
                bz: 3.1,
                v: 423.6,
                n: 5.2,
                t: 102450,
                kp: 2.3,
                dst: -12,
                ap: 7,
                f10_7: 143.2,
                s10: null,
                m10: null,
                y10: null,
              }
            : {
                id: i + 1,
                metric: ["bx", "by", "bz", "v"][i % 4],
                value: 2.513 + i,
                observed_at: "2026-09-11T09:00:00Z",
              },
        ),
        total: 1234,
        page: Number(url.searchParams.get("page") ?? 1),
        page_size: 50,
      };
    }
    await route.fulfill({ json: { success: true, data, error: null } });
  });
  return { writes, reads };
}

test("overview, chart, desktop navigation and persistent collapse", async ({
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
  await page.locator('[data-sidebar="trigger"]').click();
  await expect(
    page.locator('[data-state="collapsed"][data-collapsible="icon"]'),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.locator('[data-state="collapsed"][data-collapsible="icon"]'),
  ).toBeVisible();
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

test("observation filtering, pagination and normalized table", async ({
  page,
}) => {
  const { reads } = await mockApi(page);
  await page.goto("/dashboard/observations");
  await expect(
    page.getByText("Original measurements", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("From (UTC)").fill("2026-09-01T00:00");
  await page.getByLabel("Metric", { exact: true }).fill("bz");
  await page.getByRole("button", { name: "Apply filters" }).click();
  await expect
    .poll(() =>
      reads.some(
        (url) =>
          url.searchParams.get("metric") === "bz" &&
          url.searchParams.get("start") === "2026-09-01T00:00:00.000Z",
      ),
    )
    .toBeTruthy();
  await page.getByRole("button", { name: "Next page" }).click();
  await expect
    .poll(() => reads.some((url) => url.searchParams.get("page") === "2"))
    .toBeTruthy();
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(page.getByLabel("Metric", { exact: true })).toHaveValue("");
  await page.screenshot({
    path: "/tmp/argus-dashboard-observations.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: "Normalized", exact: true }).click();
  await expect(
    page.getByRole("columnheader", { name: "Speed", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Y10", exact: true }),
  ).toBeAttached();
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
  await expect(dialog).toHaveCSS("background-color", "rgb(17, 17, 19)");
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

test("mobile sidebar closes on navigation; wide tables remain contained", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockApi(page);
  await page.goto("/dashboard");
  await expect(
    page.getByRole("heading", { name: "Project overview" }),
  ).toBeVisible();
  await page.locator('[data-sidebar="trigger"]').click();
  const sidebar = page.getByRole("dialog");
  await expect(sidebar).toBeVisible();
  await sidebar
    .getByRole("link", { name: "Normalized data", exact: true })
    .click();
  await expect(sidebar).not.toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Speed", exact: true }),
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
  await page.getByRole("menuitem", { name: "Sign out", exact: true }).click();
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
      .locator('[data-sidebar="menu"]')
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
  await page.goto("/");
  const before = await page
    .locator("h1")
    .first()
    .evaluate((element) => {
      const css = getComputedStyle(element);
      return { fontSize: css.fontSize, color: css.color, margin: css.margin };
    });
  await page.goto("/dashboard");
  await page.getByRole("link", { name: "Public website", exact: true }).click();
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
  await expect(page.getByText("Client dashboard is coming soon")).toBeVisible();
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
