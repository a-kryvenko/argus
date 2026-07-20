
import type { ForecastPoint } from "./api";

export function prepareWindChartData(f: ForecastPoint[]) : Array<any> {
  return f.map(d => ({
    time: d.valid_time,
    median: Math.trunc(d.variables.v?.continuous?.q50 ?? 0),
    low: Math.trunc(d.variables.v?.continuous?.q10 ?? 0),
    high: Math.trunc(d.variables.v?.continuous?.q90 ?? 0),
  }));
}

export function prepareProbabilityHeatmapData(f: ForecastPoint[], variable: string) : Array<Array<Number>> {
  const riskData: Array<Array<Number>> = []
  for (let i = 0; i < f.length; i++) {
    const binary = f[i].variables[variable]?.binary ?? [];
    for (let thresholdIndex = 0; thresholdIndex < binary.length; thresholdIndex++) {
      riskData.push([
        i,
        thresholdIndex,
        Math.floor(binary[thresholdIndex].probability * 100),
      ])
    }
  }

  return riskData;
}

export const preparePlasmaHeatmapData = (f: ForecastPoint[]) => prepareProbabilityHeatmapData(f, "v");
export const prepareKpHeatmapData = (f: ForecastPoint[]) => prepareProbabilityHeatmapData(f, "kp");
