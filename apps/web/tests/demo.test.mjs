import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const source = await readFile(new URL('../src/app/demo/_lib/replay.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { parseOffset, replayTime, countdown, demoHref, selectRelease, comparisonRows, observationKnownAt, selectEventRelease, demoPageBundle } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

test('page payload keeps replay history for its product and navigation metadata without unrelated forecasts', () => {
  const release = { issue_time: '2026-01-19T19:00:00Z', available_variables: ['v'], predictions: [{ lead_hours: 1 }] };
  const density = { ...release, available_variables: ['n'] };
  const source = { releases: { speed: [release], density: [density], empty: [] }, observations: [], density_scenario: {} };
  const page = demoPageBundle(source, 'density');
  assert.deepEqual(Object.keys(page.releases), ['density']);
  assert.equal(selectRelease(page, 'density', Date.parse(release.issue_time)), density);
  assert.deepEqual(page.available_variables_by_product, { speed: ['v'], density: ['n'] });
  assert.equal(page.observations, source.observations);
  assert.equal(page.density_scenario, undefined);
  assert.equal(page.density_scenario_available, true);
  const observations = demoPageBundle(source);
  assert.deepEqual(observations.releases, {});
  assert.deepEqual(observations.available_variables_by_product, page.available_variables_by_product);
  const leo = demoPageBundle(source, undefined, true);
  assert.deepEqual(leo.releases, {});
  assert.equal(leo.density_scenario, source.density_scenario);
  assert.deepEqual(Object.keys(source.releases), ['speed', 'density', 'empty']);
  assert.ok(source.density_scenario);
});

test('offset is a bounded integer, links retain it and T0 is the precise event time', () => {
  assert.equal(parseOffset(null), -96);
  assert.equal(parseOffset('banana'), -96);
  assert.equal(parseOffset('-12.5'), -96);
  assert.equal(parseOffset('-999'), -96);
  assert.equal(parseOffset('20'), 20);
  assert.equal(parseOffset('999'), 72);
  assert.equal(parseOffset('-48'), -48);
  assert.equal(demoHref('/', -48), '/demo?offset=-48');
  assert.equal(demoHref('/forecast/dst', 0), '/demo/forecast/dst?offset=0');
  assert.equal(countdown(0), 'T0 · Event begins');
  assert.equal(countdown(-48), 'T−48 h · Until event');
  assert.equal(countdown(72), 'T+72 h · Since event');
  const bundle = { event: { starts_at: '2026-01-19T19:38:00Z' } };
  assert.equal(new Date(replayTime(bundle, -48)).toISOString(), '2026-01-17T19:38:00.000Z');
  assert.equal(new Date(replayTime(bundle, 72)).toISOString(), '2026-01-22T19:38:00.000Z');
});

test('release selection never borrows another hour or operational forecasts', () => {
  const forecast = { issue_time: '2026-01-19T19:00:00Z' };
  const bundle = { releases: { v: [forecast] } };
  assert.equal(selectRelease(bundle, 'v', Date.parse('2026-01-19T19:38:00Z')), forecast);
  assert.equal(selectRelease(bundle, 'v', Date.parse('2026-01-19T18:38:00Z')), null);
  assert.equal(selectRelease(bundle, 'v', Date.parse('2026-01-19T20:38:00Z')), null);
  assert.equal(selectRelease(bundle, 'missing', Date.parse(forecast.issue_time)), null);
});

test('comparison matches instants, keeps gaps and hides incomplete hours and future outcomes', () => {
  const now = Date.parse('2026-01-19T19:38:00Z');
  const observations = [18, 19, 20, 21].map(hour => ({ time: `2026-01-19T${hour}:00:00Z`, values: { v: hour === 21 ? null : 400 + hour } }));
  const forecast = { predictions: [20, 21].map(hour => ({ valid_time: `2026-01-19T${hour}:00:00+00:00`, lead_hours: hour - 19,
    variables: { v: { continuous: { q10: 390, q50: 410, q90: 430 } } } })) };
  const shown = comparisonRows(forecast, observations, 'v', now, true);
  assert.equal(shown[0].observed, 418);
  assert.equal(shown[1].observed, null);
  assert.equal(shown[1].actual, 419);
  assert.equal(shown.at(-2).actual, 420);
  assert.equal(shown.at(-2).median, 410);
  assert.equal(shown.at(-1).actual, null);
  const hidden = comparisonRows(forecast, observations, 'v', now, false);
  assert.ok(hidden.every(row => row.actual === null));
  assert.equal(hidden[1].observed, null);
});


test('geomagnetic and daily indices are hidden until their complete intervals close', () => {
  const time = '2026-01-19T19:00:00Z';
  assert.equal(new Date(observationKnownAt(time, 'kp')).toISOString(), '2026-01-19T21:00:00.000Z');
  assert.equal(new Date(observationKnownAt(time, 'ap')).toISOString(), '2026-01-19T21:00:00.000Z');
  assert.equal(new Date(observationKnownAt(time, 'f10_7')).toISOString(), '2026-01-20T00:00:00.000Z');
  assert.equal(new Date(observationKnownAt(time, 'dst')).toISOString(), '2026-01-19T20:00:00.000Z');
});


test('CME replay selects the exact historical issue without borrowing future releases', () => {
  const report = { issue_time: '2026-01-18T22:00:00Z', events: [{ event_id: 'known' }] };
  const bundle = { event_releases: [report] };
  assert.equal(selectEventRelease(bundle, Date.parse('2026-01-18T21:38Z')), null);
  assert.equal(selectEventRelease(bundle, Date.parse('2026-01-18T22:38Z')), report);
  assert.equal(selectEventRelease(bundle, Date.parse('2026-01-18T23:38Z')), null);
});

test('CME scenarios remain aligned with translated speed quantiles and hidden outcomes', () => {
  const now = Date.parse('2026-01-18T22:00Z');
  const time = '2026-01-18T23:00:00Z';
  const forecast = { predictions: [{ valid_time: time, lead_hours: 1,
    variables: { v: { continuous: { q10: 500, q50: 550, q90: 600 } } } }],
    cme_speed: { state: 'available', events: [{}], points: [{ valid_time: time, speed_nominal: 580,
      speed_early: 570, speed_late: 560, scenario_lower: 550, scenario_upper: 590 }] } };
  const rows = comparisonRows(forecast, [{ time, values: { v: 900 } }], 'v', now, false);
  const row = rows.find(row => row.time === time);
  assert.equal(row.median, 550);
  assert.equal(row.cmeNominal, 580);
  assert.deepEqual(row.cmeRange, [550, 590]);
  assert.equal(row.actual, null);
});
