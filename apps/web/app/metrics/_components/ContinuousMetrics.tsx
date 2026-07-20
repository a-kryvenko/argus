import type { ForecastMetrics } from "../../_utils/api";
import MetricChart from "./MetricChart";

export default function ContinuousMetrics({
  data,
  variable,
  label,
}: {
  data: ForecastMetrics;
  variable: string;
  label: string;
}) {
  const continuous = data.variables[variable]?.continuous;
  if (!continuous || continuous.by_lead_hour.length === 0) return null;

  const metricNames = Object.keys(continuous.by_lead_hour[0].values);
  return (
    <>
      {metricNames.map(metric => (
        <MetricChart
          key={metric}
          title={`${label}: ${metric.replaceAll("_", " ").toUpperCase()}`}
          labels={{ [variable]: label }}
          data={continuous.by_lead_hour.map(row => ({ [variable]: row.values[metric] }))}
        />
      ))}
    </>
  );
}
