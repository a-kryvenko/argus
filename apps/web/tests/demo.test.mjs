import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const source = await readFile(new URL('../app/demo/_lib/replay.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { parseOffset, replayTime, countdown, demoHref, selectRelease, comparisonRows, observationKnownAt } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

test('offset is a bounded integer, links retain it and T0 is the precise event time', () => {
  assert.equal(parseOffset(null), -96);
  assert.equal(parseOffset('banana'), -96);
  assert.equal(parseOffset('-12.5'), -96);
  assert.equal(parseOffset('-999'), -96);
  assert.equal(parseOffset('20'), 0);
  assert.equal(parseOffset('-48'), -48);
  assert.equal(demoHref('/', -48), '/demo?offset=-48');
  assert.equal(demoHref('/products/dst', 0), '/demo/products/dst?offset=0');
  assert.equal(countdown(0), 'T0 · Event begins');
  assert.equal(countdown(-48), 'T−48 h · Until event');
  const bundle = { event: { starts_at: '2026-01-19T19:38:00Z' } };
  assert.equal(new Date(replayTime(bundle, -48)).toISOString(), '2026-01-17T19:38:00.000Z');
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
