import { test, expect } from '@playwright/test';

const id = '00000000-0000-0000-0000-000000000001';
const profile = {
  id, revision: 1, updated_at: '2026-10-14T12:00:00Z', name: 'Satellite A', altitude_km: 400,
  inclination_deg: 51.6, mass_kg: 100, horizon_hours: 24,
  thresholds: { elevated_altitude_loss_m: 50, high_altitude_loss_m: 100 },
  modes: [{ name: 'Nominal', effective_area_m2: 1, drag_coefficient: 2.2 }], active_mode: 'Nominal', enabled: true,
  event_policy: null,
};
const envelope = (data: unknown) => ({ success: true, data, error: null });

test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/dashboard/density-advisory', route => route.fulfill({ json: envelope([]) }));
  await page.route('**/api/v1/dashboard/density-advisory/events?*', route => route.fulfill({ json: envelope({ items: [], total: 0 }) }));
  await page.route('**/api/v1/dashboard/drag-monitor/events?*', route => route.fulfill({ json: envelope({ items: [], total: 0, unacknowledged: 0 }) }));
  await page.route('**/api/v1/dashboard/me', route => route.fulfill({ json: envelope({ id: 2, username: 'member', active: true, groups: [], permissions: [] }) }));
  await page.route(`**/api/v1/dashboard/drag-monitor/${id}/chart?*`, route => route.fulfill({ json: envelope({
    run_id: new URL(route.request().url()).searchParams.get('run_id'), prepared_at: '2026-10-15T12:00:00Z',
    start_time: null, end_time: null, horizon_hours: 24, now_position: 'unavailable',
    now_lead_hours: null, elapsed_until_hours: null, max_loss_m: 110, data_status: 'unavailable',
    historical: false, thresholds: profile.thresholds,
    modes: [{ mode: 'Nominal', active: true, error: 'unavailable', points: [], elapsed: [], remaining: [],
      crossings: [], previous: [], previous_issue_time: null, previous_release_id: null, loss_change_m: null }],
  }) }));
});

test('creates a spacecraft, validates limits and preserves input values', async ({ page }) => {
  let saved = false;
  await page.route('**/api/v1/dashboard/drag-monitor', route => {
    if (route.request().method() === 'POST') {
      const { id: _id, revision: _revision, updated_at: _updated, ...body } = profile;
      void _id; void _revision; void _updated;
      expect(route.request().postDataJSON()).toEqual(body);
      saved = true;
      return route.fulfill({ status: 201, json: envelope(profile) });
    }
    return route.fulfill({ json: envelope(saved ? [{ profile, latest: null, attention: 'not_assessed', data_status: 'not_assessed', loss_change_m: null }] : []) });
  });
  await page.goto('/dashboard/risk/leo');
  await page.getByRole('button', { name: 'Experimental loss monitoring' }).click();
  await expect(page.getByRole('heading', { name: 'Drag risk', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Add spacecraft', exact: true }).click();
  await page.getByLabel('Spacecraft name', { exact: true }).fill('Satellite A');
  await page.getByLabel('Mean altitude (km)').fill('400');
  await page.getByLabel('Orbit inclination (°)').fill('51.6');
  await page.getByLabel('Spacecraft mass (kg)').fill('100');
  await page.getByLabel('Review at total altitude loss (m)').fill('50');
  await page.getByLabel('High at total altitude loss (m)').fill('40');
  await page.getByLabel('Mode 1 effective area (m²)').fill('1');
  await page.getByLabel('Mode 1 drag coefficient').fill('2.2');
  await page.getByRole('button', { name: 'Save spacecraft' }).click();
  await expect(page.locator('.dashboard [role="alert"]')).toContainText('high limit must exceed');
  expect(saved).toBe(false);
  await page.getByLabel('High at total altitude loss (m)').fill('100');
  await page.getByRole('button', { name: 'Save spacecraft' }).click();
  await expect(page.getByRole('cell', { name: 'Satellite A' })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Awaiting assessment' })).toBeVisible();
});

test('stale assessments cannot appear within limits; pause uses revision and history retains snapshots', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  let paused = false;
  const run = { id: 'run-1', profile_id: id, profile_revision: 1, profile_snapshot: profile,
    assessed_at: '2026-10-14T12:00:00Z', release_id: null,
    modes: [{ mode: 'Nominal', assessment: null, error: 'unavailable', elevated_at: null, high_at: null }] };
  await page.route('**/api/v1/dashboard/drag-monitor', route => route.fulfill({ json: envelope([{
    profile: { ...profile, enabled: !paused }, latest: run, attention: 'not_assessed', data_status: 'stale', loss_change_m: null,
  }]) }));
  await page.route(`**/api/v1/dashboard/drag-monitor/${id}?revision=1`, route => {
    expect(route.request().method()).toBe('PUT');
    expect(route.request().postDataJSON().enabled).toBe(false);
    expect(route.request().postDataJSON().owner_id).toBeUndefined();
    paused = true;
    return route.fulfill({ json: envelope({ ...profile, revision: 2, enabled: false }) });
  });
  await page.route(`**/api/v1/dashboard/drag-monitor/${id}/history`, route => route.fulfill({ json: envelope([run]) }));
  await page.goto('/dashboard/risk/leo');
  await page.getByRole('button', { name: 'Experimental loss monitoring' }).click();
  await expect(page.getByRole('cell', { name: 'Stale atmosphere' })).toBeVisible();
  await expect(page.getByText('Within limits', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Details', exact: true }).click();
  await expect(page.getByText(/Saved assessment · Profile v1/)).toBeVisible();
  await expect(page.getByText(/No risk category assigned/)).toBeVisible();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Resume', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Assess now' })).toBeDisabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('current assessment explains attention, mode results and threshold crossing time', async ({ page }) => {
  const start = '2026-10-14T12:00:00Z';
  const end = '2026-10-15T12:00:00Z';
  const assessment = {
    start_time: start, end_time: end, computed_at: start,
    estimated_altitude_loss_m: 100, delta_v_loss_m_s: 0.1, drag_risk: 'high',
    risk_reason: 'Estimated loss 100 m over 24 h; high user limit 100 m reached.',
    source: { density_model: 'JB2008', release_id: id, observed_at: start,
      geomagnetic_observed_at: start, driver_mode: 'observed_persistence' },
    assumptions: ['Circular orbit; observed drivers held fixed.'],
    predictions: Array.from({ length: 25 }, (_, i) => ({
      valid_time: new Date(Date.parse(start) + i * 3600000).toISOString(), lead_hours: i,
      mean_density_kg_m3: 1e-12, mean_drag_accel_m_s2: 1e-6,
      delta_v_loss_m_s: i / 240, estimated_altitude_loss_m: i * 100 / 24,
    })),
  };
  const run = { id: 'current-run', profile_id: id, profile_revision: 1, profile_snapshot: profile,
    assessed_at: start, release_id: id, modes: [{ mode: 'Nominal', assessment, error: null,
      elevated_at: '2026-10-15T00:00:00Z', high_at: end }] };
  const points = assessment.predictions.map(point => ({ lead_hours: point.lead_hours, valid_time: point.valid_time, altitude_loss_m: point.estimated_altitude_loss_m }));
  const previous = points.map(point => ({ ...point, valid_time: new Date(Date.parse(point.valid_time)-3600000).toISOString(), altitude_loss_m: point.altitude_loss_m*0.8 }));
  await page.route(`**/api/v1/dashboard/drag-monitor/${id}/chart?*`, route => route.fulfill({ json: envelope({
    run_id: 'current-run', prepared_at: '2026-10-14T15:30:00Z', start_time: start, end_time: end,
    horizon_hours: 24, now_position: 'within', now_lead_hours: 3.5, elapsed_until_hours: 3.5, max_loss_m: 110,
    data_status: 'current', historical: false, thresholds: profile.thresholds,
    modes: [
      { mode: 'Nominal', active: true, error: null, points, elapsed: points.slice(0, 4), remaining: points.slice(3),
        crossings: [{ level: 'elevated', lead_hours: 12, valid_time: '2026-10-15T00:00:00Z', altitude_loss_m: 50, state: 'upcoming' },
          { level: 'high', lead_hours: 24, valid_time: end, altitude_loss_m: 100, state: 'upcoming' }],
        previous, previous_issue_time: '2026-10-14T11:00:00Z', previous_release_id: 'previous-release', loss_change_m: 20 },
      { mode: 'Low drag', active: false, error: null, points: points.map(point => ({ ...point, altitude_loss_m: point.altitude_loss_m*0.4 })),
        elapsed: [], remaining: points.map(point => ({ ...point, altitude_loss_m: point.altitude_loss_m*0.4 })),
        crossings: [], previous: [], previous_issue_time: null, previous_release_id: null, loss_change_m: null },
      { mode: 'Safe mode', active: false, error: 'outside_model_domain', points: [], elapsed: [], remaining: [],
        crossings: [], previous: [], previous_issue_time: null, previous_release_id: null, loss_change_m: null },
    ],
  }) }));
  await page.route('**/api/v1/dashboard/drag-monitor', route => route.fulfill({ json: envelope([{
    profile, latest: run, data_status: 'current', attention: 'high', loss_change_m: 20,
  }]) }));
  await page.route(`**/api/v1/dashboard/drag-monitor/${id}/history`, route => route.fulfill({ json: envelope([run]) }));
  await page.goto('/dashboard/risk/leo');
  await page.getByRole('button', { name: 'Experimental loss monitoring' }).click();
  await expect(page.getByRole('cell', { name: 'High limit reached' })).toBeVisible();
  await expect(page.getByRole('cell', { name: '+20 m', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Details', exact: true }).click();
  await expect(page.getByText('100 m altitude loss · 0.1 m/s drag Δv')).toBeVisible();
  await expect(page.getByText(/High limit: 15\/10\/2026, 12:00:00 UTC/)).toBeVisible();
  await expect(page.getByRole('img', { name: /Cumulative altitude loss/ }).locator('canvas')).toBeVisible();
  const panel = page.getByRole('region', { name: 'Operator drag chart' });
  await expect(panel.getByText('Within the selected interval', { exact: false })).toBeVisible();
  await expect(panel.getByRole('heading', { name: 'Low drag', exact: true })).toBeVisible();
  await expect(panel.getByText('Outside model limits · No curve available')).toBeVisible();
  await expect(panel.getByText('Change: +20 m', { exact: false })).toBeVisible();
  await panel.getByText('Chart data', { exact: true }).click();
  await expect(panel.getByRole('cell', { name: 'Previous', exact: true })).toHaveCount(25);
  await panel.getByRole('button', { name: 'Hide previous releases' }).click();
  await expect(panel.getByRole('cell', { name: 'Previous', exact: true })).toHaveCount(0);
  await panel.getByText('Chart data', { exact: true }).click();
  await page.getByText('Source and assumptions', { exact: true }).click();
  await expect(page.getByText('Circular orbit; observed drivers held fixed.')).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/drag-monitor-current-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('event review persists and opens the exact historical assessment beyond the history page', async ({ page }) => {
  const eventId = '00000000-0000-0000-0000-000000000010';
  const oldId = '00000000-0000-0000-0000-000000000011';
  const newId = '00000000-0000-0000-0000-000000000012';
  const start = '2026-10-14T12:00:00Z';
  const assessment = {
    start_time: start, end_time: '2026-10-15T12:00:00Z', computed_at: start,
    estimated_altitude_loss_m: 120, delta_v_loss_m_s: 0.1, drag_risk: 'high', risk_reason: 'High user limit reached.',
    source: { density_model: 'JB2008', release_id: id, observed_at: start, geomagnetic_observed_at: start, driver_mode: 'observed_persistence' },
    assumptions: ['Circular orbit'], predictions: Array.from({ length: 25 }, (_, i) => ({
      valid_time: new Date(Date.parse(start)+i*3600000).toISOString(), lead_hours: i,
      mean_density_kg_m3: 1e-12, mean_drag_accel_m_s2: 1e-6, delta_v_loss_m_s: i/240, estimated_altitude_loss_m: i*5,
    })),
  };
  const old = { id: oldId, profile_id: id, profile_revision: 1, profile_snapshot: profile, assessed_at: start,
    release_id: id, modes: [{ mode: 'Nominal', assessment, error: null, elevated_at: '2026-10-14T22:00:00Z', high_at: '2026-10-15T08:00:00Z' }] };
  const current = { ...old, id: newId, profile_revision: 2, profile_snapshot: { ...profile, mass_kg: 200 }, assessed_at: '2026-10-15T06:00:00Z' };
  let acknowledged = false;
  const event = () => ({ id: eventId, profile_id: id, profile_revision: 1, spacecraft_name: 'Satellite A', active_mode: 'Nominal',
    kind: 'risk_high', severity: 'high', detected_at: start, expected_at: '2026-10-15T08:00:00Z', run_id: oldId,
    release_id: id, comparison_run_id: null, message: 'Estimated loss reaches the high user limit.',
    values: { estimated_loss_m: 120, high_limit_m: 100, horizon_hours: 24 },
    acknowledged_at: acknowledged ? '2026-10-15T07:00:00Z' : null, acknowledged_by: acknowledged ? 2 : null });
  await page.route('**/api/v1/dashboard/drag-monitor', route => route.fulfill({ json: envelope([{
    profile: { ...profile, revision: 2, mass_kg: 200 }, latest: current, data_status: 'current', attention: 'high', loss_change_m: null,
  }]) }));
  await page.route('**/api/v1/dashboard/drag-monitor/events?*', route => {
    const unread = new URL(route.request().url()).searchParams.get('unacknowledged') === 'true';
    const entries = unread && acknowledged ? [] : [event()];
    return route.fulfill({ json: envelope({ items: entries, total: entries.length, unacknowledged: acknowledged ? 0 : 1 }) });
  });
  await page.route(`**/api/v1/dashboard/drag-monitor/events/${eventId}/ack`, route => {
    expect(route.request().method()).toBe('POST'); acknowledged = true;
    return route.fulfill({ json: envelope(event()) });
  });
  await page.route(`**/api/v1/dashboard/drag-monitor/${id}/runs/${oldId}`, route => route.fulfill({ json: envelope(old) }));
  await page.route(`**/api/v1/dashboard/drag-monitor/${id}/history`, route => route.fulfill({ json: envelope([current]) }));
  await page.goto('/dashboard/risk/leo');
  await page.getByRole('button', { name: 'Experimental loss monitoring' }).click();
  const entry = page.getByRole('article', { name: 'Satellite A: High limit reached' });
  await expect(entry).toBeVisible();
  await entry.getByRole('button', { name: 'Open assessment' }).click();
  await expect(page.getByText(/Saved assessment · Profile v1/)).toBeVisible();
  await expect(page.getByText('120 m altitude loss · 0.1 m/s drag Δv')).toBeVisible();
  await entry.getByRole('button', { name: 'Mark reviewed' }).click();
  await expect(entry.getByRole('button', { name: 'Reviewed', exact: true })).toBeDisabled();
  await expect(page.getByText('0 unreviewed · 1 event')).toBeVisible();
  await page.getByLabel('Unreviewed only').check();
  await expect(page.getByText('No events', { exact: true })).toBeVisible();
});
