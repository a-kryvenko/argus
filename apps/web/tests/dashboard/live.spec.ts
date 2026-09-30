import { test, expect, type Page } from '@playwright/test';

async function observations(page: Page, options: { stale?: boolean; unavailable?: boolean; historyFailure?: boolean } = {}) {
  const now = new Date().toISOString();
  const before = new Date(Date.now() - 3600000).toISOString();
  const requests: URL[] = [];
  const values = { bx: 1.4, by: 2.1, bz: -3.6, bt: 5.8, v: 420, n: 6.2, t: 108000 };
  const solarWind = Object.fromEntries(Object.entries(values).map(([metric, value]) => [metric, {
    latest: { observed_at: now, received_at: now, value, spacecraft: 'DSCOVR', quality: metric === 'by' ? 'flagged' : 'unverified' },
    age_seconds: options.stale ? 900 : 0, status: options.stale ? 'stale' : 'fresh', stale_after_seconds: 600,
  }]));
  const geomagnetic = Object.fromEntries(['kp', 'dst'].map(metric => [metric, {
    latest: { interval_start: before, interval_end: now, interval_status: 'completed', value: metric === 'kp' ? 3 : -20,
      quality: 'unverified', received_at: now, station_count: null },
    lag_seconds: 0, status: 'fresh', stale_after_seconds: metric === 'kp' ? 14400 : 7200, data_status: metric === 'kp' ? 'estimated' : 'realtime',
  }]));
  await page.route('**/api/v1/public/observations/**', async route => {
    const url = new URL(route.request().url());
    expect(url.searchParams.has('meta')).toBe(false);
    if (url.searchParams.has('from')) requests.push(url);
    if (options.unavailable || options.historyFailure && url.searchParams.has('from')) {
      await route.fulfill({ status: 503, json: { success: false, data: null, error: { message: 'Not ready' } } });
      return;
    }
    let data;
    if (url.pathname.endsWith('/summary')) {
      data = { generated_at: now, solar_wind: solarWind, geomagnetic,
        changes_1h: { v: { status: 'available', value: 10, as_of: now }, n: { status: 'available', value: .4, as_of: now },
          bt: { status: 'available', value: .8, as_of: now }, bz: { status: 'available', value: -1.2, as_of: now } },
        southward_bz: { status: 'lower_bound', value: 12, as_of: now, reason: 'onset_before_available_window' } };
    } else if (url.searchParams.has('from')) {
      const from = url.searchParams.get('from')!, to = url.searchParams.get('to')!;
      const start = Date.parse(from), end = Date.parse(to);
      const resolution = end - start > 7 * 86400000 ? 3600 : end - start > 86400000 ? 300 : 60;
      if (url.pathname.endsWith('/solar-wind/history')) {
        const count = Math.floor((end - start) / (resolution * 1000));
        const coverage = { expected_slots: count, usable_slots: count - 2, missing_slots: 2, invalid_slots: 0,
          percent: Math.round((count - 2) / count * 10000) / 100, resolution_seconds: 60, evaluated_to: to, basis: 'sample_timestamps',
          gaps: [{ from, to: new Date(start + resolution * 2000).toISOString(), reason: 'missing', slots: 2 }] };
        const series = Object.fromEntries(Object.entries(solarWind).map(([metric, series]) => [metric, {
          points: Array.from({ length: count - 2 }, (_, i) => {
            const amplitude = metric === 'v' ? 20 : metric === 't' ? 10000 : 2;
            const value = series.latest.value + amplitude * (Math.sin(i / 64) + .18 * Math.sin(i / 8));
            return { ...series.latest, quality: 'unverified', observed_at: new Date(start + (i + 2) * resolution * 1000).toISOString(),
              value, ...(resolution > 60 ? { min: value - amplitude * .3, max: value + amplitude * .3, count: resolution / 60,
                expected_count: resolution / 60, coverage_percent: 100, recalculation_pending: false, source_changes: 0, last_spacecraft: 'DSCOVR' } : {}) };
          }), coverage,
        }]));
        data = { from, to, resolution_seconds: resolution, series };
      } else {
        data = { from, to, series: Object.fromEntries(Object.entries(geomagnetic).map(([metric, series]) => {
          const interval = metric === 'kp' ? 10800000 : 3600000;
          const left = Math.floor(start / interval) * interval;
          const points = Array.from({ length: Math.ceil((end - left) / interval) }, (_, i) => ({
            ...series.latest, value: metric === 'kp' ? 2 + (i % 4) / 3 : -20 + Math.sin(i / 4) * 15,
            interval_start: new Date(left + i * interval).toISOString(), interval_end: new Date(left + (i + 1) * interval).toISOString(),
          }));
          return [metric, { points, data_status: series.data_status, coverage: {
            expected_slots: points.length, usable_slots: points.length, missing_slots: 0, invalid_slots: 0,
            percent: 100, resolution_seconds: interval / 1000, evaluated_to: to, basis: 'overlapping_intervals', gaps: [],
          } }];
        })) };
      }
    } else if (url.pathname.endsWith('/status')) {
      data = { generated_at: now, status: 'ok', sources: Object.fromEntries(['solar_wind_mag', 'kp', 'dst'].map(source_id => [source_id, {
        source_id, label: source_id === 'solar_wind_mag' ? 'Solar wind magnetic field' : source_id.toUpperCase(),
        source_url: 'https://www.spaceweather.gov/', status: 'ok', poll_seconds: 60,
        last_attempt_at: now, last_completed_at: now, last_response_at: now, last_success_at: now,
        last_error_at: null, last_error_message: null, last_error_code: null, consecutive_failures: 0,
        latest_observation_at: now, latest_interval_end: now,
      }])) };
    } else {
      data = { points: [] };
    }
    await route.fulfill({ json: { success: true, data, error: null } });
  });
  return requests;
}

test('workspace renders compact observations and inspects quality without losing chart context', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await observations(page);
  await page.goto('/live');
  const summary = page.getByRole('region', { name: 'Observation summary' });
  const inspector = page.getByRole('complementary', { name: 'Selected observation' });
  await expect(summary).toContainText('420 km/s');
  await expect(summary).toContainText('1h change: +10 km/s');
  await expect(summary).toContainText('-3.6 nT');
  await expect(summary).toContainText('Negative for at least 12 min');
  await expect(summary).toContainText('-20 nT');
  await expect(inspector.getByRole('heading', { name: 'Bz · GSM', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Real-time Dst history' })).toContainText('Real-time Dst (nT)');
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  await expect(page.getByRole('group', { name: 'History period' })).toHaveCount(1);
  await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Live observations' })).toHaveAttribute('aria-current', 'page');
  await page.addStyleTag({ content: 'nextjs-portal { display: none; }' });
  await page.screenshot({ path: 'test-results/live-workspace-desktop.png', fullPage: true });
  await summary.getByRole('button', { name: 'Inspect Real-time Dst' }).click();
  await expect(inspector).toContainText('WDC Kyoto via NOAA SWPC');
  await inspector.getByLabel('Selected measurement').selectOption('by');
  await expect(inspector).toContainText('Provider quality flag');
  await expect(inspector).toContainText('excluded from chart lines and derived trends');
  await expect(page.getByRole('region', { name: 'By, nT', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('one period control updates both histories with exactly the same UTC bounds', async ({ page }) => {
  const requests = await observations(page);
  await page.goto('/live');
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  await page.getByRole('button', { name: '3 days', exact: true }).click();
  await expect(page.getByRole('button', { name: '3 days', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('5-minute means', { exact: true })).toBeVisible();
  await expect.poll(() => requests.filter(url => Date.parse(url.searchParams.get('to')!) - Date.parse(url.searchParams.get('from')!) === 72 * 3600000).length).toBeGreaterThanOrEqual(2);
  const latest = requests.filter(url => Date.parse(url.searchParams.get('to')!) - Date.parse(url.searchParams.get('from')!) === 72 * 3600000).slice(-2);
  expect(new Set(latest.map(url => url.pathname)).size).toBe(2);
  expect(latest[0].searchParams.get('from')).toBe(latest[1].searchParams.get('from'));
  expect(latest[0].searchParams.get('to')).toBe(latest[1].searchParams.get('to'));
});

test('mobile keeps navigation, shared controls and an expandable inspector without overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await observations(page);
  await page.goto('/live');
  const inspector = page.getByRole('complementary', { name: 'Selected observation' });
  await page.getByText('Resources', { exact: true }).click();
  await expect(page.getByRole('navigation', { name: 'Mobile resources' }).getByRole('link', { name: 'Help & methodology' })).toBeVisible();
  await page.getByText('Resources', { exact: true }).click();
  await expect(inspector.getByLabel('Selected measurement')).toBeHidden();
  await page.getByRole('region', { name: 'Observation summary' }).getByRole('button', { name: 'Inspect Solar wind speed' }).click();
  await expect(inspector.getByLabel('Selected measurement')).toBeVisible();
  await expect(inspector.getByLabel('Selected measurement')).toHaveValue('v');
  await expect(inspector).toContainText('420 km/s');
  await inspector.getByRole('button', { name: 'Solar wind speed' }).click();
  await expect(inspector.getByLabel('Selected measurement')).toBeHidden();
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.addStyleTag({ content: 'nextjs-portal { display: none; }' });
  await page.screenshot({ path: 'test-results/live-workspace-mobile.png', fullPage: true });
});

test('delayed values do not produce an apparently current southward duration or trend', async ({ page }) => {
  await observations(page, { stale: true, historyFailure: true });
  await page.goto('/live');
  const summary = page.getByRole('region', { name: 'Observation summary' });
  await expect(summary.getByRole('button', { name: 'Inspect Bz', exact: true })).toContainText('Delayed');
  await expect(summary).toContainText('Duration unavailable');
  await expect(summary).not.toContainText('Negative for');
  await expect(summary).not.toContainText('1h change:');
  await expect(page.getByRole('alert').filter({ hasText: 'Solar wind history' })).toBeVisible();
  await expect(page.getByRole('alert').filter({ hasText: 'Index history' })).toBeVisible();
});

test('unavailable data stays explicit and never turns into zero readings', async ({ page }) => {
  await observations(page, { unavailable: true });
  await page.goto('/live');
  const summary = page.getByRole('region', { name: 'Observation summary' });
  await expect(summary.getByRole('alert')).toBeVisible();
  await expect(summary.getByRole('button', { name: 'Inspect Solar wind speed' })).toContainText('— km/s');
  await expect(summary.getByRole('button', { name: 'Inspect Solar wind speed' })).toContainText('Unavailable');
  await expect(page.getByRole('heading', { name: 'Live observations', exact: true })).toBeVisible();
});

test('failed refresh keeps the last history, but switching period never relabels old data', async ({ page }) => {
  await page.clock.install({ time: new Date() });
  const options = { historyFailure: false };
  const requests = await observations(page, options);
  await page.goto('/live');
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  const originalCount = requests.length;
  options.historyFailure = true;
  await page.clock.fastForward(60000);
  await expect.poll(() => requests.length).toBeGreaterThan(originalCount);
  await expect(page.getByRole('alert').filter({ hasText: 'Solar wind history' })).toContainText('Showing the last response for this period');
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  await page.getByRole('button', { name: '3 days', exact: true }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Solar wind history' })).toContainText('No history available for this period');
  await expect(page.locator('.recharts-line-curve')).toHaveCount(0);
});
