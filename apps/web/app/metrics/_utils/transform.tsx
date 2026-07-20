import type { ForecastMetrics } from "../../_utils/api";

type Metric = "brier_score" | "roc_auc" | "average_precision" | "reliability";

export default function transformMetrics(
  data: ForecastMetrics,
  variable: string,
  metric: Metric
) {
  const series = data.variables[variable]?.binary ?? [];

  const maxLength = Math.max(0, ...series.map(item => item.by_lead_hour.length));

  return Array.from({ length: maxLength }, (_, i) => {
    const row: Record<string, any> = {};

    for (const item of series) {
      row[String(item.threshold)] = item.by_lead_hour[i]?.[metric] ?? null;
    }

    return row;
  });
}
