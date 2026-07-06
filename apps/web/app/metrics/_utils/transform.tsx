type Metric = "brier" | "roc_auc" | "avg_precision" | "reliability";

export default function transformMetrics(
  data: Record<string, Record<string, any>[]>,
  metric: Metric
) {
  const keys = Object.keys(data);

  const maxLength = Math.max(...keys.map(k => data[k].length));

  return Array.from({ length: maxLength }, (_, i) => {
    const row: Record<string, any> = {};

    for (const key of keys) {
      row[key] = data[key][i]?.[metric] ?? null;
    }

    return row;
  });
}