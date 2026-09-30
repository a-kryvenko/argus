import { URL } from "node:url";
import { Buffer } from "node:buffer";
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
const source = await readFile(
  new URL("../app/_utils/forecast.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext },
}).outputText;
const { formatForecastTime, probabilityData, quantileData, forecastWindow, formatProbability } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);

test('horizon filters relative lead hours without changing issue time or filling missing variables', () => {
  const forecast = { issue_time: '2020-01-01T00:00:00Z', horizon_hours: 48,
    predictions: [{ lead_hours: 1, variables: { bt: {} } }, { lead_hours: 24, variables: {} },
      { lead_hours: 30, variables: { bs: {} } }] };
  const view = forecastWindow(forecast, 24);
  assert.equal(view.issue_time, forecast.issue_time);
  assert.deepEqual(view.predictions.map(point => point.lead_hours), [1, 24]);
  assert.deepEqual(view.predictions[1].variables, {});
  assert.equal(forecast.predictions.length, 3);
  assert.equal(forecastWindow(forecast, null).predictions.length, 3);
});

test('probabilities distinguish missing, zero and very small positive values', () => {
  assert.equal(formatProbability(undefined), '—');
  assert.equal(formatProbability(null), '—');
  assert.equal(formatProbability(0), '0%');
  assert.equal(formatProbability(1), '100%');
  assert.equal(formatProbability(.725), '72.5%');
  assert.equal(formatProbability(.0005), '<0.1%');
});

test("forecast timestamps preserve historical dates and normalize offsets to UTC", () => {
  assert.equal(
    formatForecastTime("2025-12-31T23:00:00-02:00"),
    "01 Jan 2026, 01:00 UTC",
  );
  assert.equal(formatForecastTime("invalid"), "Time unavailable");
});
test("missing quantiles leave a gap at the original valid time rather than zero or removing the hour", () => {
  const forecast = {
    predictions: [
      {
        valid_time: "2026-01-01T01:00:00Z",
        variables: {
          v: { continuous: { q10: 390.2, q50: 400.5, q90: 420.8 } },
        },
      },
      { valid_time: "2026-01-01T02:00:00Z", variables: {} },
    ],
  };
  assert.deepEqual(quantileData(forecast, "v"), [
    {
      time: forecast.predictions[0].valid_time,
      low: 390.2,
      median: 400.5,
      high: 420.8,
    },
    {
      time: forecast.predictions[1].valid_time,
      low: null,
      median: null,
      high: null,
    },
  ]);
});
test("heatmap maps unordered thresholds to labelled rows and preserves zero probability", () => {
  const forecast = {
    predictions: [
      {
        variables: {
          v: {
            binary: [
              { threshold: 600, probability: 0, operator: "gte" },
              { threshold: 450, probability: 0.725, operator: "gte" },
            ],
          },
        },
      },
      { variables: {} },
    ],
  };
  assert.deepEqual(probabilityData(forecast, "v", [450, 500, 600]), [
    [0, 0, 73],
    [0, 2, 0],
  ]);
});

test("heatmap accepts the current API payload without an operator", () => {
  const forecast = {
    predictions: [{ variables: { kp: { binary: [
      { threshold: 4, probability: 0.725 },
      { threshold: 5, probability: 0 },
      { threshold: 6, probability: 0.1, operator: "lt" },
    ] } } }],
  };
  assert.deepEqual(probabilityData(forecast, "kp", [4, 5, 6]), [
    [0, 0, 73],
    [0, 1, 0],
  ]);
});
