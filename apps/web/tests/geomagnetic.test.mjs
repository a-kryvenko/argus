import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const source = await readFile(new URL('../app/live/geomagnetic.ts', import.meta.url), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { intervalBounds } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const point = { interval_start: '2026-09-07T12:00:00Z', interval_end: '2026-09-07T15:00:00Z', value: 3.33, quality: 'unverified' };
const start = Date.parse('2026-09-07T13:00:00Z'), end = Date.parse('2026-09-07T16:00:00Z');
test('partially visible Kp intervals use true bounds rather than synthetic hourly points', () => {
  assert.deepEqual(intervalBounds(point, start, end), [start, Date.parse(point.interval_end)]);
  assert.equal(point.value, 3.33);
});
test('gaps and flagged values remain blank', () => {
  assert.equal(intervalBounds({ ...point, value: null }, start, end), null);
  assert.equal(intervalBounds({ ...point, quality: 'flagged' }, start, end), null);
  assert.equal(intervalBounds(point, Date.parse(point.interval_end), end), null);
});
