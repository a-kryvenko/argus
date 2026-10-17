import { test, expect } from '@playwright/test';

const id = '00000000-0000-0000-0000-000000000011';
const runId = '00000000-0000-0000-0000-000000000012';
const issue = '2026-10-15T12:00:00Z';
const trajectory = { source: 'Uploaded ephemeris A', generated_at: issue, valid_until: '2026-10-16T12:00:00Z', coordinates: 'geocentric_spherical', points: [
  { time: '2026-10-15T13:00:00Z', altitude_km: 400, latitude_deg: 0, longitude_deg: 10 },
  { time: '2026-10-15T13:10:00Z', altitude_km: 401, latitude_deg: 20, longitude_deg: 30 },
] };
const input = { name: 'Known trajectory A', apoapsis_km: 420, periapsis_km: 400, inclination_deg: 51.6, weight_kg: null, effective_area_m2: null, policy: { review_ratio: 2, high_ratio: 4 } };
const profile = { ...input, enabled: true, id, revision: 1, updated_at: issue };
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

test('density advisories are default and save spacecraft without trajectory or reference inputs', async ({ page }) => {
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
  await expect(page.getByRole('heading', { name: 'Drag risk' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Experimental loss monitoring' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Calculator', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Add spacecraft', exact: true }).click();
  await page.getByLabel('Spacecraft name', { exact: true }).fill(input.name);
  await expect(page.getByLabel('Apoapsis (km)', { exact: true })).toHaveAttribute('required', '');
  await expect(page.getByLabel('Weight (kg, optional)', { exact: true })).not.toHaveAttribute('required');
  await page.getByLabel('Apoapsis (km)', { exact: true }).fill('420');
  await page.getByLabel('Periapsis (km)', { exact: true }).fill('400');
  await page.getByLabel('Inclination (°)', { exact: true }).fill('51.6');
  await page.getByLabel('High density ratio', { exact: true }).fill('2');
  await page.getByRole('button', { name: 'Save spacecraft' }).click();
  await expect(page.getByText('High density ratio must exceed review density ratio.', { exact: true })).toBeVisible();
  expect(saved).toBe(false);
  await page.getByLabel('High density ratio', { exact: true }).fill('4');
  await expect(page.getByRole('checkbox')).toHaveCount(0);
  for (const label of ['Upload trajectory JSON', 'Trajectory and reference JSON', 'Maximum sample gap (minutes)', 'Reference source']) {
    await expect(page.getByLabel(label, { exact: true })).toHaveCount(0);
  }
  await page.getByRole('button', { name: 'Save spacecraft' }).click();
  await expect(page.getByRole('button', { name: input.name, exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Warning', exact: true })).toBeVisible();
});

test('requires a dashboard session', async ({ page }) => {
  await page.route('**/api/v1/dashboard/me', route => route.fulfill({ status: 401,
    json: { success: false, data: null, error: { message: 'Sign in' } } }));
  await page.goto('/dashboard/risk/leo');
  await expect(page).toHaveURL(/\/dashboard\/login/);
});

test('compact spacecraft table explains warnings and only marks complete current coverage OK', async ({ page }) => {
  const cases = [
    { name: 'All clear', attention: 'within_reference', data_status: 'current', cme_status: 'current', status: 'OK', explanation: 'No warnings.' },
    { name: 'Density increase', attention: 'review', data_status: 'current', cme_status: 'current', status: 'Warning', details: ['Forecast density reaches 3.2× the reference.'], explanation: 'Forecast density reaches 3.2× the reference.' },
    { name: 'High density', attention: 'high', data_status: 'current', cme_status: 'current', status: 'Warning', explanation: 'Density exceeds the high warning threshold.' },
    { name: 'Incoming CME', attention: 'watch', data_status: 'current', cme_status: 'current', status: 'Warning', explanation: 'A predicted CME arrival falls within the forecast period.' },
    { name: 'Old density', attention: 'not_assessed', data_status: 'stale', cme_status: 'current', status: 'Warning', explanation: 'Density forecast is out of date.' },
    { name: 'No CME data', attention: 'within_reference', data_status: 'current', cme_status: 'unavailable', status: 'Warning', explanation: 'CME forecast is unavailable.' },
    { name: 'Incomplete density', attention: 'not_assessed', data_status: 'partial', cme_status: 'current', status: 'Warning', explanation: 'Density or reference coverage is incomplete.' },
    { name: 'Paused satellite', attention: 'not_assessed', data_status: 'paused', cme_status: 'current', status: 'Warning', explanation: 'Monitoring is paused.' },
  ];
  await page.route('**/api/v1/dashboard/density-advisory', route => route.fulfill({ json: envelope(cases.map((item, i) => ({
    ...item, profile: { ...profile, id: `${id}-${i}`, name: item.name }, latest: { id: runId, profile_revision: 1, computed_at: issue },
  }))) }));
  await page.goto('/dashboard/risk/leo');
  const table = page.getByRole('table', { name: 'Spacecraft status' });
  await expect(table.getByRole('columnheader')).toHaveText(['Spacecraft', 'Status', 'Details', 'Actions']);
  for (const item of cases) {
    const row = table.getByRole('row').filter({ has: page.getByRole('button', { name: item.name, exact: true }) });
    await expect(row.getByRole('cell').nth(0)).toContainText('400km/420km/51.6°');
    await expect(row.getByRole('cell').nth(1)).toHaveText(item.status);
    await expect(row.getByRole('cell').nth(2)).toContainText(item.explanation);
    await expect(row.getByRole('cell').nth(2).getByRole('button')).toHaveCount(0);
  }
});

test('updates spacecraft properties, validates apsides and preserves optional values', async ({ page }) => {
  const updated = { ...profile, weight_kg: 125, effective_area_m2: 2.5 };
  let saved = false;
  await page.route('**/api/v1/dashboard/density-advisory', route => route.fulfill({ json: envelope([
    { profile: updated, latest: null, attention: 'not_assessed', data_status: 'unavailable' },
  ]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}`, route => route.fulfill({ json: envelope(updated) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}?revision=1`, route => {
    expect(route.request().method()).toBe('PUT');
    expect(route.request().postDataJSON()).toEqual({ ...input, apoapsis_km: 450, weight_kg: 150, effective_area_m2: null });
    saved = true;
    return route.fulfill({ json: envelope({ ...updated, revision: 2 }) });
  });
  await page.goto('/dashboard/risk/leo');
  await page.getByRole('button', { name: 'Edit', exact: true }).click();
  await expect(page.getByLabel('Apoapsis (km)', { exact: true })).toHaveValue('420');
  await expect(page.getByLabel('Inclination (°)', { exact: true })).toHaveValue('51.6');
  await expect(page.getByLabel('Weight (kg, optional)', { exact: true })).toHaveValue('125');
  await expect(page.getByLabel('Effective area (m², optional)', { exact: true })).toHaveValue('2.5');
  await page.getByLabel('Apoapsis (km)', { exact: true }).fill('399');
  await page.getByRole('button', { name: 'Save spacecraft' }).click();
  await expect(page.getByText('Apoapsis must be greater than or equal to periapsis.')).toBeVisible();
  expect(saved).toBe(false);
  await page.getByLabel('Apoapsis (km)', { exact: true }).fill('450');
  await page.getByLabel('Weight (kg, optional)', { exact: true }).fill('0');
  await page.getByRole('button', { name: 'Save spacecraft' }).click();
  await expect(page.getByText('Weight and effective area must be positive when supplied.')).toBeVisible();
  await page.getByLabel('Weight (kg, optional)', { exact: true }).fill('150');
  await page.getByLabel('Effective area (m², optional)', { exact: true }).fill('');
  await page.getByRole('button', { name: 'Save spacecraft' }).click();
  await expect(page.getByRole('button', { name: 'Save spacecraft' })).toHaveCount(0);
  expect(saved).toBe(true);
});

test('deletion can be cancelled, keeps spacecraft on failure and clears deleted forecast state', async ({ page }) => {
  let removed = false;
  let allowDelete = false;
  let deleteCalls = 0;
  const other = { ...profile, id: 'other-spacecraft', name: 'Satellite B' };
  await page.route('**/api/v1/dashboard/density-advisory', route => route.fulfill({ json: envelope([
    ...(!removed ? [{ profile, latest: { id: runId, profile_revision: 1, computed_at: issue }, attention: 'watch', data_status: 'partial', cme_status: 'current' }] : []),
    { profile: other, latest: null, attention: 'not_assessed', data_status: 'unavailable', cme_status: 'unavailable' },
  ]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}/history`, route => route.fulfill({ json: envelope([{ id: runId, profile_revision: 1, computed_at: issue }]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}/runs/${runId}`, route => route.fulfill({ json: envelope(run) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}?revision=1`, route => {
    expect(route.request().method()).toBe('DELETE');
    deleteCalls++;
    if (!allowDelete) return route.fulfill({ status: 409, json: { success: false, data: null, error: { message: 'Profile changed. Reload and try again.' } } });
    removed = true;
    return route.fulfill({ json: envelope({ deleted: true }) });
  });
  await page.goto('/dashboard/risk/leo');
  const row = page.getByRole('table', { name: 'Spacecraft status' }).getByRole('row').filter({ has: page.getByRole('button', { name: input.name, exact: true }) });
  await row.getByRole('button', { name: 'Delete', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Cancel', exact: true }).click();
  expect(deleteCalls).toBe(0);
  await page.getByRole('button', { name: input.name, exact: true }).click();
  await page.getByRole('button', { name: 'View forecast', exact: true }).click();
  await expect(page.getByRole('heading', { name: /Saved density forecast/ })).toBeVisible();
  await row.getByRole('button', { name: 'Delete', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('button', { name: 'Delete spacecraft', exact: true }).click();
  await expect(dialog.getByRole('alert')).toContainText('Profile changed');
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(row).toHaveCount(1);
  allowDelete = true;
  await row.getByRole('button', { name: 'Delete', exact: true }).click();
  await dialog.getByRole('button', { name: 'Delete spacecraft', exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await expect(row).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Satellite B', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: /Saved density forecast/ })).toHaveCount(0);
  await expect(page.getByLabel('Saved forecast history')).toHaveCount(0);
  expect(deleteCalls).toBe(2);
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
  await expect(page.getByText('A predicted CME arrival falls within the forecast period.', { exact: true })).toBeVisible();
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
  await expect(page.getByText('Density forecast is out of date.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Mark reviewed' }).click();
  await expect(page.getByText(/^Reviewed /)).toBeVisible();
  await expect(page.getByRole('cell', { name: 'Warning', exact: true })).toBeVisible();
});

test('published background warning explains magnitude, sampled span, peak location and missing CME context', async ({ page }) => {
  const background = { label: 'JB2008 fixed-geomagnetic reference', source: 'Prophet density release background-1',
    prepared_at: issue, kind: 'model_geomagnetic_reference', method: 'jb2008_fixed_ap4_v1' };
  const published = { ...run, assessment: { ...run.assessment, attention: 'review', data_status: 'current',
    source: { ...run.assessment.source, reference: background, cme_status: 'unavailable', assessment_basis: 'orbit_envelope', trajectory_source: null, trajectory_generated_at: null },
    intervals: [{ start: trajectory.points[0].time, end: trajectory.points[1].time, attention: 'review',
      peak_density_kg_m3: 3.2e-12, peak_ratio: 3.2, sample_count: 2, sampled_duration_minutes: 10,
      peak_location: { ...trajectory.points[1], longitude_deg: null }, reasons: [
        { code: 'density_ratio', message: 'Forecast density reaches 3.2× the reference.', source_id: 'background-1' },
        { code: 'reference_definition', message: 'Reference: JB2008 fixed-geomagnetic reference.', source_id: 'background-1' },
        { code: 'cme_context_unavailable', message: 'CME context is stale or unavailable; this warning is based on density alone.', source_id: null },
      ] }],
  } };
  await page.route('**/api/v1/dashboard/density-advisory', route => route.fulfill({ json: envelope([{ profile, latest: { id: runId, profile_revision: 1, computed_at: issue }, attention: 'review', data_status: 'current', cme_status: 'unavailable' }]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}/history`, route => route.fulfill({ json: envelope([{ id: runId, profile_revision: 1, computed_at: issue }]) }));
  await page.route(`**/api/v1/dashboard/density-advisory/${id}/runs/${runId}`, route => route.fulfill({ json: envelope(published) }));
  await page.goto('/dashboard/risk/leo');
  await page.getByRole('button', { name: input.name, exact: true }).click();
  await page.getByRole('button', { name: 'View forecast', exact: true }).click();
  await expect(page.getByText(/Model background · prepared/)).toBeVisible();
  await expect(page.getByText('Forecast density reaches 3.2× the reference.')).toBeVisible();
  await expect(page.getByText(/Warning forecast steps span 10 minutes/)).toBeVisible();
  await expect(page.getByText(/Envelope maximum at.*401.0 km.*latitude 20.00/)).toBeVisible();
  await expect(page.getByText(/this warning is based on density alone/)).toBeVisible();
  await expect(page.getByText('Orbital envelope forecast values', { exact: true })).toBeVisible();
  await expect(page.getByText('Trajectory', { exact: true })).toHaveCount(0);
});
