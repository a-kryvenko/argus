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
const { formatForecastTime, probabilityData, quantileData } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);

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
