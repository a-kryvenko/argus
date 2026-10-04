import { URL } from 'node:url';
import { Buffer } from 'node:buffer';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const source = await readFile(new URL('../app/metrics/_utils/transform.tsx', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { binaryRows, leadHours, continuousKeys, chartRows, metricNumber, metricTitle } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

test('threshold scores align by actual lead hour across unordered and unequal series', () => {
  const variable = { continuous: { by_lead_hour: [{ lead_hours: 48, values: { mae: 4 } }] }, binary: [
    { threshold: 500, by_lead_hour: [{ lead_hours: 6, brier_score: .3 }, { lead_hours: 1, brier_score: 0 }] },
    { threshold: 600, by_lead_hour: [{ lead_hours: 3, brier_score: null }, { lead_hours: 6, brier_score: .2 }] },
  ] };
  assert.deepEqual(binaryRows(variable, 'brier_score'), [
    { lead_hours: 1, values: { 500: 0, 600: null } },
    { lead_hours: 3, values: { 500: null, 600: null } },
    { lead_hours: 6, values: { 500: .3, 600: .2 } },
  ]);
  assert.deepEqual(leadHours(variable), [1, 3, 6, 48]);
});
test('continuous metric choices include keys absent from the first lead', () => {
  assert.deepEqual(continuousKeys({ continuous: { by_lead_hour: [{ values: { n: 20, mae: null } }, { values: { rmse: 4, mae: 2 } }] } }), ['n', 'mae', 'rmse']);
});
test('chart preserves true lead times and breaks omitted hours without mutating rows', () => {
  const rows = [{ lead_hours: 6, values: { score: .2 } }, { lead_hours: 3, values: { score: 0 } }];
  assert.deepEqual(chartRows(rows), [rows[1], { lead_hours: 4, values: {} }, rows[0]]);
  assert.equal(rows[0].lead_hours, 6);
});
test('missing and nonfinite scores differ from zero and negative skill', () => {
  assert.equal(metricNumber(null), '—');
  assert.equal(metricNumber(NaN), '—');
  assert.equal(metricNumber(0), '0');
  assert.equal(metricNumber(-.25), '-0.25');
  assert.deepEqual(leadHours(), []);
  assert.deepEqual(chartRows([]), []);
});

test('metric precision distinguishes tiny nonzero values and retains exact titles', () => {
  assert.equal(metricNumber(15.1433), '15');
  assert.equal(metricNumber(19.538), '20');
  assert.equal(metricNumber(1), '1');
  assert.equal(metricNumber(.123456), '0.123');
  assert.equal(metricNumber(.001), '0.001');
  assert.equal(metricNumber(.000004949486147261768), '<0.001');
  assert.equal(metricTitle(.000004949486147261768), '0.000004949486147261768');
  assert.equal(metricNumber(0), '0');
  assert.equal(metricNumber(-.00001), '>-0.001');
  assert.equal(metricNumber(-12.7), '-13');
  assert.equal(metricTitle(null), undefined);
});
