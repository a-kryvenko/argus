import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const source = await readFile(new URL('../app/live/solarWind.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { chartPoints } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
const point = (time, value, spacecraft = 'A', quality = 'unverified') => ({
  observed_at: new Date(time).toISOString(), value, spacecraft, quality,
});

test('omitted minutes break a line instead of interpolating', () => {
  const rows = chartPoints({ series: { bz: { points: [point(0, -2), point(180000, -4)] } } });
  assert.equal(rows.length, 3);
  assert.equal(rows[1].bz, undefined);
});

test('a spacecraft change breaks the line, including adjacent minutes', () => {
  const rows = chartPoints({ series: { bz: { points: [point(0, -2), point(60000, -4, 'B')] } } });
  assert.equal(rows.length, 3);
  assert.equal(rows[1].bz, undefined);
});

test('provider-flagged numeric samples are excluded from lines', () => {
  const rows = chartPoints({ series: { bz: { points: [point(0, -2, 'A', 'flagged')] } } });
  assert.equal(rows[0].bz, null);
});

test('plasma spacecraft changes do not introduce gaps into the magnetic chart', () => {
  const history = { series: {
    bz: { points: [point(0, -2), point(60000, -4)] },
    v: { points: [point(0, 400), point(60000, 410, 'B')] },
  } };
  const rows = chartPoints(history, ['bz']);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map(row => row.bz), [-2, -4]);
});

test('aggregate chart preserves tooltip statistics and breaks missing buckets', () => {
  const history = { resolution_seconds: 300, series: { bz: { points: [
    { observed_at: '2026-09-01T00:00:00Z', value: -2, min: -8, max: 3, coverage_percent: 60 },
    { observed_at: '2026-09-01T00:10:00Z', value: 1, min: -1, max: 4, coverage_percent: 100 },
  ] } } };
  const rows = chartPoints(history, ['bz']);
  assert.equal(rows.length, 3);
  assert.equal(rows[1].bz, undefined);
  assert.equal(rows[0].bz, -2);
  assert.equal(rows[0].bz_min, -8);
  assert.equal(rows[0].bz_max, 3);
  assert.equal(rows[0].bz_coverage, 60);
});

test('adjacent aggregate means form a line until the spacecraft changes', () => {
  const samples = [0, 300000, 600000].map((time, index) => ({
    ...point(time, index, index === 2 ? 'B' : 'A'),
    source_changes: 0,
  }));
  const rows = chartPoints({ resolution_seconds: 300, series: { bz: { points: samples } } }, ['bz']);
  assert.deepEqual(rows.map(row => row.bz), [0, 1, undefined, 2]);
});
