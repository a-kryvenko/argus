import { test, expect } from '@playwright/test';

const id = '00000000-0000-0000-0000-000000000011';
const runId = '00000000-0000-0000-0000-000000000012';
const issue = '2026-10-15T12:00:00Z';
const trajectory = { source: 'Uploaded ephemeris A', generated_at: issue, valid_until: '2026-10-16T12:00:00Z', coordinates: 'geocentric_spherical', points: [
  { time: '2026-10-15T13:00:00Z', altitude_km: 400, latitude_deg: 0, longitude_deg: 10 },
  { time: '2026-10-15T13:10:00Z', altitude_km: 401, latitude_deg: 20, longitude_deg: 30 },
] };
const input = { name: 'Known trajectory A', trajectory, reference: null, reference_mode: 'auto', policy: { review_ratio: 2, high_ratio: 4, max_sample_gap_minutes: 15 }, enabled: true };
const profile = { ...input, id, revision: 1, updated_at: issue };
const run = { id: runId, profile_id: id, profile_revision: 1, profile_snapshot: input, assessment: {
  computed_at: issue, attention: 'watch', data_status: 'partial',
  source: { density_model: 'JB2008', density_release_id: id, density_issue_time: issue, density_observed_at: issue,
    cme_release_id: id, cme_issue_time: issue, cme_status: 'current', trajectory_source: trajectory.source, trajectory_generated_at: issue, reference: null },
  samples: trajectory.points.map(p => ({ ...p, density_kg_m3: 1e-12, reference_density_kg_m3: null, ratio: null, attention: 'not_assessed',
    reasons: [{ code: 'reference_missing', message: 'Relative density anomaly is not assessed without a reference.', source_id: null }] })),
  intervals: [{ start: trajectory.points[0].time, end: trajectory.points[0].time, attention: 'watch', peak_ratio: null, peak_density_kg_m3: null, sample_count: 0,
    reasons: [{ code: 'cme_arrival', message: 'Published CME Earth-arrival prediction overlaps this interval; storm strength is unknown.', source_id: 'CME-A' }] }],
  limitations: ['Density is a longitude mean; longitude is not resolved.', 'CME does not numerically modify density.'],
} };
const envelope = (data: unknown) => ({ success: true, data, error: null });

test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/dashboard/me', route => route.fulfill({ json: envelope({ id: 2, username: 'member', active: true, groups: [], permissions: [] }) }));
  await page.route('**/api/v1/dashboard/density-advisory/events?*', route => route.fulfill({ json: envelope({ items: [], total: 0 }) }));
});

test('density advisories are default and accept supplied ephemerides without spacecraft physics', async ({ page }) => {
  let saved = false;
  await page.route('**/api/v1/dashboard/density-advisory', route => {
    if (route.request().method() === 'POST') {
      expect(route.request().postDataJSON()).toEqual(input);
      saved = true;
      return route.fulfill({ status: 201, json: envelope(profile) });
    }
    return route.fulfill({ json: envelope(saved ? [{ profile, latest: null, attention: 'not_assessed', data_status: 'unavailable' }] : []) });
  });
  await page.goto('/dashboard/risk/leo');
  await expect(page.getByRole('heading', { name: 'Atmospheric density risk' })).toBeVisible();
  await expect(page.getByLabel('Spacecraft mass (kg)')).toHaveCount(0);
  await page.getByRole('button', { name: 'Add trajectory', exact: true }).click();
  await page.getByLabel('Spacecraft name', { exact: true }).fill(input.name);
  await page.getByLabel('Trajectory and reference JSON').fill('{');
  await page.getByRole('button', { name: 'Save trajectory' }).click();
  await expect(page.locator('.dashboard [role="alert"]')).toBeVisible();
  await expect(page.getByLabel('Spacecraft name', { exact: true })).toHaveValue(input.name);
  await page.getByLabel('Trajectory and reference JSON').fill(JSON.stringify({ trajectory, reference: null }));
  await page.getByRole('button', { name: 'Save trajectory' }).click();
  await expect(page.getByRole('cell', { name: input.name })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Not assessed', exact: true })).toBeVisible();
});

test('CME-only watch explains missing reference and opens immutable event source outside history page', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route('**/api/v1/dashboard/density-advisory', route => route.fulfill({ json: envelope([{ profile, latest: run, attention: 'watch', data_status: 'partial' }]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}/history`, route => route.fulfill({ json: envelope([]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}/runs/${runId}`, route => route.fulfill({ json: envelope(run) }));
  await page.route('**/api/v1/dashboard/density-advisory/events?*', route => route.fulfill({ json: envelope({ total: 1, items: [{
    id: 'event-1', profile_id: id, run_id: runId, detected_at: issue, kind: 'attention_changed', attention: 'watch', message: 'CME watch detected', acknowledged_at: null,
  }] }) }));
  await page.goto('/dashboard/risk/leo');
  await expect(page.getByRole('cell', { name: 'CME watch', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Open source forecast' }).click();
  await expect(page.getByRole('heading', { name: /Saved density forecast/ })).toBeVisible();
  await expect(page.getByText('Unavailable — relative anomaly not assessed')).toBeVisible();
  await expect(page.getByText(/storm strength is unknown/)).toBeVisible();
  await expect(page.getByText('Source: CME-A')).toBeVisible();
  await page.getByText('Trajectory sample values and coverage').click();
  await expect(page.getByText('Relative density anomaly is not assessed without a reference.').first()).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Below configured ratios' })).toHaveCount(0);
});

test('stale saved warning does not become current attention and review does not change it', async ({ page }) => {
  let reviewed = false;
  await page.route('**/api/v1/dashboard/density-advisory', route => route.fulfill({ json: envelope([{ profile, latest: run, attention: 'not_assessed', data_status: 'stale' }]) }));
  await page.route('**/api/v1/dashboard/density-advisory/events?*', route => route.fulfill({ json: envelope({ total: 1, items: [{
    id: id, profile_id: id, run_id: runId, detected_at: issue, kind: 'data_status_changed', attention: 'not_assessed', message: 'Density stale', acknowledged_at: reviewed ? issue : null,
  }] }) }));
  await page.route(`**/api/v1/dashboard/density-advisory/events/${id}/ack`, route => {
    expect(route.request().method()).toBe('POST'); reviewed = true;
    return route.fulfill({ json: envelope({}) });
  });
  await page.goto('/dashboard/risk/leo');
  await expect(page.getByRole('cell', { name: 'Stale density', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Mark reviewed' }).click();
  await expect(page.getByText(/^Reviewed /)).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Not assessed', exact: true })).toBeVisible();
});

test('published background warning explains magnitude, sampled span, peak location and missing CME context', async ({ page }) => {
  const background = { label: 'JB2008 fixed-geomagnetic reference', source: 'Prophet density release background-1',
    prepared_at: issue, kind: 'model_geomagnetic_reference', method: 'jb2008_fixed_ap4_v1' };
  const published = { ...run, assessment: { ...run.assessment, attention: 'review', data_status: 'current',
    source: { ...run.assessment.source, reference: background, cme_status: 'unavailable' },
    intervals: [{ start: trajectory.points[0].time, end: trajectory.points[1].time, attention: 'review',
      peak_density_kg_m3: 3.2e-12, peak_ratio: 3.2, sample_count: 2, sampled_duration_minutes: 10,
      peak_location: { ...trajectory.points[1] }, reasons: [
        { code: 'density_ratio', message: 'Forecast density reaches 3.2× the reference.', source_id: 'background-1' },
        { code: 'reference_definition', message: 'Reference: JB2008 fixed-geomagnetic reference.', source_id: 'background-1' },
        { code: 'cme_context_unavailable', message: 'CME context is stale or unavailable; this warning is based on density alone.', source_id: null },
      ] }],
  } };
  await page.route('**/api/v1/dashboard/density-advisory', route => route.fulfill({ json: envelope([{ profile, latest: { id: runId, profile_revision: 1, computed_at: issue }, attention: 'review', data_status: 'current', cme_status: 'unavailable' }]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}/history`, route => route.fulfill({ json: envelope([{ id: runId, profile_revision: 1, computed_at: issue }]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}/runs/${runId}`, route => route.fulfill({ json: envelope(published) }));
  await page.goto('/dashboard/risk/leo');
  await page.getByRole('button', { name: 'Details', exact: true }).click();
  await expect(page.getByText(/Model background · prepared/)).toBeVisible();
  await expect(page.getByText('Forecast density reaches 3.2× the reference.')).toBeVisible();
  await expect(page.getByText(/Warning samples span 10 minutes/)).toBeVisible();
  await expect(page.getByText(/Peak ratio at.*401.0 km.*latitude 20.00/)).toBeVisible();
  await expect(page.getByText(/this warning is based on density alone/)).toBeVisible();
});
