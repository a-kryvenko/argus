import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const source = await readFile(new URL('../app/demo/_lib/density.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { densityRelease, densityRows, visibleDensityJournal } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);

const issue = Date.parse('2026-01-19T19:00Z');
const iso = offset => new Date(issue + offset * 3600000).toISOString();
const release = { issue_time: iso(0), predictions: [{ time: iso(1), ratio: 3 }, { time: iso(2), ratio: null }] };
const scenario = { reference: { density_kg_m3: 2 }, releases: { jb2008_fixed: [release], dtm2020_fixed: [] },
  observations: [-1, 0, 1, 2].map(offset => ({ time: iso(offset), density_kg_m3: offset === 2 ? null : 4 })),
  journal: [-1, 0, 1].map(offset => ({ variant: 'jb2008_fixed', issue_time: iso(offset) })) };

test('density uses only the exact historical issue and variant', () => {
  assert.equal(densityRelease(scenario, 'jb2008_fixed', issue + 38 * 60000), release);
  assert.equal(densityRelease(scenario, 'jb2008_fixed', issue + 3600000), null);
  assert.equal(densityRelease(scenario, 'dtm2020_fixed', issue), null);
});

test('hiding outcomes retains forecasts and missing samples without leaking future truth', () => {
  const rows = densityRows(scenario, release, issue, false);
  assert.equal(rows.find(row => row.timestamp === issue).observed, 2);
  assert.equal(rows.find(row => row.timestamp === issue + 3600000).forecast, 3);
  assert.ok(rows.every(row => row.actual === null));
  assert.equal(rows.at(-1).forecast, null);
  assert.equal(densityRows(scenario, release, issue, true).find(row => row.timestamp === issue + 3600000).actual, 2);
  assert.equal(densityRows(scenario, release, issue, true).at(-1).actual, null);
});

test('revision history cannot reveal future issuances', () => {
  assert.deepEqual(visibleDensityJournal(scenario, 'jb2008_fixed', issue).map(row => row.issue_time), [iso(0), iso(-1)]);
  assert.deepEqual(visibleDensityJournal(scenario, 'dtm2020_fixed', issue), []);
});
