import { captureScreenshot } from "./screenshot";
import { test, expect, type Page } from '@playwright/test';

test('live metadata and dataset description are available without JavaScript', async ({ request }) => {
  const response = await request.get('/live');
  expect(response.ok()).toBe(true);
  const html = await response.text();
  expect(html).toContain('<title>Live Solar Wind, Bz &amp; Kp Index | Argus SunWatch</title>');
  expect(html).toContain('rel="canonical" href="https://argussun.com/live"');
  expect(html).toContain('property="og:type" content="website"');
  expect(html).toContain('name="twitter:card" content="summary_large_image"');
  expect(html).toContain('Live space weather</h1>');
  const jsonLd = html.match(/<script type="application\/ld\+json">(.*?)<\/script>/s);
  expect(jsonLd).not.toBeNull();
  const dataset = JSON.parse(jsonLd![1]);
  expect(dataset['@type']).toBe('Dataset');
  expect(dataset.variableMeasured.map((variable: { propertyID: string }) => variable.propertyID)).toEqual(['v', 'n', 'bz', 'bt', 'kp', 'dst', 'f10_7']);
  expect(dataset.distribution[0].contentUrl).toBe('https://argussun.com/api/v1/observations');
  expect(dataset).not.toHaveProperty('dateModified');
  const image = await request.get('/live/opengraph-image');
  expect(image.ok()).toBe(true);
  expect(image.headers()['content-type']).toContain('image/png');
  const png = await image.body();
  expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([1200, 630]);
  expect(await (await request.get('/robots.txt')).text()).toContain('Sitemap: https://argussun.com/sitemap.xml');
  expect(await (await request.get('/sitemap.xml')).text()).toContain('<loc>https://argussun.com/live</loc>');
});

async function observations(page: Page, options: { stale?: boolean; unavailable?: boolean; historyFailure?: boolean } = {}) {
  const now = new Date().toISOString();
  const before = new Date(Date.now() - 3600000).toISOString();
  const requests: URL[] = [];
  const values = { bx: 1.4, by: 2.1, bz: -3.6, bt: 5.8, v: 420, n: 6.2, t: 108000 };
  const solarWind = Object.fromEntries(Object.entries(values).map(([metric, value]) => [metric, {
    latest: { observed_at: now, received_at: now, value, quality: metric === 'by' ? 'flagged' : undefined },
    age_seconds: options.stale ? 900 : 0, status: options.stale ? 'stale' : 'fresh', stale_after_seconds: 600,
  }]));
  const geomagnetic = Object.fromEntries(['kp', 'dst'].map(metric => [metric, {
    latest: { interval_start: before, interval_end: now, interval_status: 'completed', value: metric === 'kp' ? 3 : -20,
      quality: undefined, received_at: now },
    lag_seconds: 0, status: 'fresh', stale_after_seconds: metric === 'kp' ? 14400 : 7200, data_status: metric === 'kp' ? 'estimated' : 'realtime',
  }]));
  await page.route('**/api/v1/observations**', async route => {
    const url = new URL(route.request().url());
    expect(url.searchParams.has('meta')).toBe(false);
    if (url.searchParams.has('from')) requests.push(url);
    if (options.unavailable || options.historyFailure && url.searchParams.has('from')) {
      await route.fulfill({ status: 503, json: { success: false, data: null, error: { message: 'Not ready' } } });
      return;
    }
    let data;
    if (!url.pathname.endsWith('/history')) {
      expect(url.searchParams.get('trends')).toBe('true');
      data = {
        f10_7: { observed_at: new Date(Date.now() - 24 * 3600000).toISOString(), value: 145.2 },
        changes_1h: { v: 10, n: .4, bt: .8, bz: -1.2 },
        ...Object.fromEntries(Object.entries(solarWind).map(([metric, { latest }]) => [metric, {
          observed_at: options.stale ? new Date(Date.now() - 900000).toISOString() : latest.observed_at,
          value: latest.value, quality: latest.quality,
        }])),
        ...Object.fromEntries(Object.entries(geomagnetic).map(([metric, { latest }]) => [metric, {
          interval_start: latest.interval_start, interval_end: latest.interval_end, value: latest.value, quality: latest.quality,
        }])),
      };
    } else if (url.searchParams.has('from')) {
      const from = url.searchParams.get('from')!, to = url.searchParams.get('to')!;
      const start = Date.parse(from), end = Date.parse(to);
      const resolution = end - start > 86400000 ? 3600 : end - start > 6 * 3600000 ? 300 : 60;
      const count = Math.floor((end - start) / (resolution * 1000));
      const coverage = { expected_slots: count, usable_slots: count - 2, missing_slots: 2, invalid_slots: 0,
        percent: Math.round((count - 2) / count * 10000) / 100, resolution_seconds: 60, evaluated_to: to, basis: 'sample_timestamps',
        gaps: [{ from, to: new Date(start + resolution * 2000).toISOString(), reason: 'missing', slots: 2 }] };
      const windSeries = Object.fromEntries(Object.entries(solarWind).map(([metric, series]) => [metric, {
        points: Array.from({ length: count - 2 }, (_, i) => {
          const amplitude = metric === 'v' ? 20 : metric === 't' ? 10000 : 2;
          const value = series.latest.value + amplitude * (Math.sin(i / 64) + .18 * Math.sin(i / 8));
          return { ...series.latest, quality: undefined, observed_at: new Date(start + (i + 2) * resolution * 1000).toISOString(),
            value, ...(resolution > 60 ? { min: value - amplitude * .3, max: value + amplitude * .3, count: resolution / 60,
              expected_count: resolution / 60, coverage_percent: 100, recalculation_pending: false } : {}) };
        }), coverage,
      }]));
      const indexSeries = Object.fromEntries(Object.entries(geomagnetic).map(([metric, series]) => {
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
      }));
      data = { from, to, resolution_seconds: resolution, series: { ...windSeries, ...indexSeries } };
    } else {
      data = { points: [] };
    }
    await route.fulfill({ json: { success: true, data, error: null } });
  });
  return requests;
}

test('workspace renders compact observations and opens details from summary cards', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await observations(page);
  await page.goto('/live');
  const summary = page.getByRole('region', { name: 'Observation summary' });
  const inspector = page.getByRole('region', { name: 'Selected observation' });
  await expect(summary).toContainText('420 km/s');
  await expect(summary).toContainText('1h change: +10 km/s');
  await expect(summary.getByRole('button', { name: 'Inspect Magnetic field · Bz / Bt' })).toContainText('-3.6');
  await expect(summary).toContainText('-20 nT');
  await expect(summary.getByRole('button')).toHaveCount(6);
  await expect(summary.getByRole('button', { name: 'Inspect F10.7', exact: true })).toContainText('145.2 sfu');
  await expect(summary.getByRole('button', { name: 'Inspect F10.7', exact: true })).toContainText('Recent');
  await expect(inspector).toBeVisible();
  const magneticChart = page.getByRole('region', { name: 'Magnetic field · Bz / Bt, nT', exact: true });
  await expect(magneticChart.locator('.recharts-line-curve')).toHaveCount(2);
  await expect(magneticChart).toContainText('Bz · GSM');
  await expect(magneticChart).toContainText('Bt · total');
  await expect(page.getByRole('region', { name: 'Real-time Dst history' })).toContainText('Real-time Dst (nT)');
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  await expect(page.getByRole('group', { name: 'History period' })).toHaveCount(1);
  await expect(page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Observations' })).toHaveAttribute('aria-current', 'page');
  await captureScreenshot(page, { path: 'test-results/live-workspace-desktop.png', fullPage: true });
  await summary.getByRole('button', { name: 'Inspect Magnetic field · Bz / Bt', exact: true }).click();
  await expect(inspector).toContainText('north–south magnetic field');
  await expect(inspector.getByRole('combobox')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Details', exact: true })).toHaveCount(0);
  await expect(summary).not.toContainText('SOUTHWARD Bz');
  await expect(inspector).not.toContainText('Selected period');
  expect(await inspector.evaluate(element => !!element.parentElement?.querySelector('[aria-label="Observation summary"]'))).toBe(true);
  await expect(page.getByRole('region', { name: 'Solar wind at L1', exact: true }).getByRole('button')).toHaveCount(0);
  await captureScreenshot(page, { path: 'test-results/live-workspace-inspector.png', fullPage: true });
  await summary.getByRole('button', { name: 'Inspect Real-time Dst' }).click();
  await expect(inspector).toContainText('WDC Kyoto via NOAA SWPC');
  await summary.getByRole('button', { name: 'Inspect Solar wind speed' }).click();
  await expect(inspector).toContainText('Solar wind speed');
  await expect(inspector).not.toContainText('420 km/s');
  await summary.getByRole('button', { name: 'Inspect F10.7', exact: true }).click();
  await expect(inspector).toContainText('Solar radio flux');
  await expect(inspector).toContainText('48 h');
  await expect(inspector.getByRole('link', { name: 'Measurement data & metadata' })).toHaveAttribute('href', '/api/v1/observations?meta=true&observations=f10_7');
  expect(errors).toEqual([]);
});

for (const days of [3, 7, 30]) {
test(`${days} days uses hourly wind history with one shared request`, async ({ page }) => {
  await page.clock.install({ time: new Date('2026-10-19T12:00:10Z') });
  const requests = await observations(page);
  await page.goto('/live');
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  await page.getByRole('button', { name: `${days} days`, exact: true }).click();
  await expect(page.getByRole('button', { name: `${days} days`, exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Hourly means', { exact: true })).toBeVisible();
  const periodRequests = requests.filter(url => Date.parse(url.searchParams.get('to')!) - Date.parse(url.searchParams.get('from')!) === days * 24 * 3600000);
  expect(periodRequests).toHaveLength(1);
  expect(periodRequests[0].searchParams.get('observations')).toBe('bx,by,bz,bt,v,n,t,kp,dst');
  const dstChart = page.getByRole('group', { name: 'Real-time Dst: hourly segments, UTC' });
  await expect(dstChart).toBeVisible();
  await expect(dstChart.locator('g[tabindex="0"]')).toHaveCount(days * 24);
});
}

test('mobile keeps navigation, shared controls and inline details without overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await observations(page);
  await page.goto('/live');
  const inspector = page.getByRole('region', { name: 'Selected observation' });
  await page.getByText('Resources', { exact: true }).click();
  await expect(page.getByRole('navigation', { name: 'Mobile resources' }).getByRole('link', { name: 'Help & methodology' })).toBeVisible();
  await page.getByText('Resources', { exact: true }).click();
  await expect(inspector).toBeVisible();
  await page.getByRole('region', { name: 'Observation summary' }).getByRole('button', { name: 'Inspect Solar wind speed' }).click();
  await expect(inspector).toContainText('Solar wind speed');
  await expect(inspector.getByRole('combobox')).toHaveCount(0);
  await expect(inspector).not.toContainText('420 km/s');
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await captureScreenshot(page, { path: 'test-results/live-workspace-mobile.png', fullPage: true });
});

test('delayed measurements stay explicit without trends or duration', async ({ page }) => {
  await observations(page, { stale: true, historyFailure: true });
  await page.goto('/live');
  const summary = page.getByRole('region', { name: 'Observation summary' });
  await expect(summary.getByRole('button', { name: 'Inspect Magnetic field · Bz / Bt', exact: true })).toContainText('Delayed');
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
  await expect(page.getByRole('heading', { name: 'Live space weather', exact: true })).toBeVisible();
});

test('failed refresh keeps the last history, but switching period never relabels old data', async ({ page }) => {
  await page.clock.install({ time: new Date() });
  const options = { historyFailure: false };
  const requests = await observations(page, options);
  await page.goto('/live');
  const windCurves = page.getByRole('region', { name: 'Solar wind at L1', exact: true }).locator('.recharts-line-curve');
  await expect(windCurves.first()).toBeVisible();
  const originalCount = requests.length;
  options.historyFailure = true;
  await page.clock.fastForward(60000);
  await expect.poll(() => requests.length).toBeGreaterThan(originalCount);
  await expect(page.getByRole('alert').filter({ hasText: 'Solar wind history' })).toContainText('Showing the last response for this period');
  await expect(windCurves.first()).toBeVisible();
  await page.getByRole('button', { name: '3 days', exact: true }).click();
  await expect(page.getByRole('alert').filter({ hasText: 'Solar wind history' })).toContainText('No history available for this period');
  await expect(windCurves).toHaveCount(0);
});


test('initial load sends one snapshot and one history request; hourly indices load on demand', async ({ page }) => {
  await page.clock.install({ time: new Date('2026-10-19T12:00:10Z') });
  const requests: URL[] = [];
  page.on('request', request => {
    const url = new URL(request.url());
    if (url.pathname.startsWith('/api/v1/observations')) requests.push(url);
  });
  await observations(page);
  await page.goto('/live');
  await expect(page.getByRole('region', { name: 'Observation summary' })).toContainText('420 km/s');
  await expect(page.locator('.recharts-line-curve').first()).toBeVisible();
  expect(requests.filter(url => url.pathname === '/api/v1/observations')).toHaveLength(1);
  expect(requests.filter(url => url.pathname.endsWith('/history'))).toHaveLength(1);
  expect(requests).toHaveLength(2);

  const summary = page.getByRole('region', { name: 'Observation summary' });
  await expect(summary.getByRole('button', { name: 'Inspect Solar wind speed' })).toHaveAttribute('aria-pressed', 'true');
  await summary.getByRole('button', { name: 'Inspect Real-time Dst' }).click();
  await expect(page.getByRole('region', { name: 'Selected observation' })).toContainText('WDC Kyoto via NOAA SWPC');
  await expect(summary.getByRole('button', { name: 'Inspect Real-time Dst' })).toHaveAttribute('aria-pressed', 'true');
  expect(requests).toHaveLength(2);

  await page.getByText('Additional hourly indices', { exact: false }).click();
  await expect(page.getByRole('heading', { name: 'Hourly indices', exact: true })).toBeVisible();
  await expect.poll(() => requests.filter(url => url.searchParams.get('observations') === 'ap,f10_7,s10,m10,y10').length).toBe(1);
  expect(requests).toHaveLength(3);
});
