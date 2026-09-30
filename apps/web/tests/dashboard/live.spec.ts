import { test, expect } from '@playwright/test';

test('live views render compact observations with fixed units, quality and coverage', async ({ page }) => {
  const now = new Date().toISOString();
  const before = new Date(Date.now() - 3600000).toISOString();
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const values = { bx: 1, by: 2, bz: -3, bt: 5, v: 420, n: 6, t: 100000 };
  const solarWind = Object.fromEntries(Object.entries(values).map(([metric, value]) => [metric, {
    latest: { observed_at: now, received_at: now, value, spacecraft: 'DSCOVR', quality: metric === 'by' ? 'flagged' : 'unverified' },
    age_seconds: 0, status: 'fresh', stale_after_seconds: 600,
  }]));
  const geomagnetic = Object.fromEntries(['kp', 'dst'].map(metric => [metric, {
    latest: { interval_start: before, interval_end: now, interval_status: 'completed', value: metric === 'kp' ? 3 : -20,
      quality: 'unverified', received_at: now, station_count: null },
    lag_seconds: 0, status: 'fresh', stale_after_seconds: 14400, data_status: metric === 'kp' ? 'estimated' : 'realtime',
  }]));
  const coverage = { expected_slots: 60, usable_slots: 59, missing_slots: 1, invalid_slots: 0,
    percent: 98.33, resolution_seconds: 60, evaluated_to: now, basis: 'sample_timestamps',
    gaps: [{ from: before, to: now, reason: 'missing', slots: 1 }] };
  await page.route('**/api/v1/public/observations/**', async route => {
    const url = new URL(route.request().url());
    expect(url.searchParams.has('meta')).toBe(false);
    let data;
    if (url.pathname.endsWith('/summary')) {
      data = { generated_at: now, solar_wind: solarWind, geomagnetic,
        changes_1h: { v: { status: 'available', value: 10, as_of: now }, bz: { status: 'available', value: -1, as_of: now } },
        southward_bz: { status: 'lower_bound', value: 12, as_of: now, reason: 'onset_before_available_window' } };
    } else if (url.pathname.endsWith('/solar-wind/history')) {
      data = { from: before, to: now, resolution_seconds: 60, series: Object.fromEntries(
        Object.entries(solarWind).map(([metric, series]) => [metric, { points: [series.latest], coverage }])) };
    } else if (url.pathname.endsWith('/geomagnetic/history')) {
      data = { from: before, to: now, series: Object.fromEntries(
        Object.entries(geomagnetic).map(([metric, series]) => [metric, { points: [series.latest], coverage,
          data_status: series.data_status }])) };
    } else if (url.pathname.endsWith('/status')) {
      data = { generated_at: now, status: 'ok', sources: {} };
    } else {
      data = { points: [] };
    }
    await route.fulfill({ json: { success: true, data, error: null } });
  });
  await page.goto('/live');
  const summary = page.getByRole('region', { name: 'Observation summary' });
  await expect(summary).toContainText('420 km/s');
  await expect(summary).toContainText('1h change: +10 km/s');
  await expect(summary).toContainText('-3 nT');
  await expect(summary).toContainText('Negative for at least 12 min');
  await expect(summary).toContainText('-20 nT');
  await page.getByText('Solar wind measurement details', { exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Bz · GSM', exact: true })).toBeVisible();
  await expect(page.getByText('DSCOVR · Provider quality flag', { exact: true })).toBeVisible();
  await page.getByText('Geomagnetic measurement details', { exact: true }).click();
  await expect(page.getByText('WDC Kyoto via NOAA SWPC', { exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Real-time Dst history' })).toContainText('Real-time Dst (nT)');
  await expect(page.getByText(/98.33%/).first()).toBeVisible();
  expect(errors).toEqual([]);
});
