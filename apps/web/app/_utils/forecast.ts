import type { Forecast } from "./api";

export function formatProbability(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  if (value > 0 && value < .001) return '<0.1%';
  return `${(value * 100).toLocaleString('en-US', { maximumFractionDigits: 1 })}%`;
}

// Horizons are relative to the release's issue time, not the current clock.
export function forecastWindow(forecast: Forecast, hours: number | null): Forecast {
  return { ...forecast, predictions: hours == null ? forecast.predictions :
    forecast.predictions.filter(point => point.lead_hours <= hours) };
}

export function formatForecastTime(value: string): string {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "Time unavailable";
  return (
    new Intl.DateTimeFormat("en-GB", {
      timeZone: "UTC",
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23",
    }).format(date) + " UTC"
  );
}

export function quantileData(forecast: Forecast, key: string) {
  return forecast.predictions.map((point) => {
    const values = point.variables[key]?.continuous;
    return {
      time: point.valid_time,
      low: values?.q10 ?? null,
      median: values?.q50 ?? null,
      high: values?.q90 ?? null,
    };
  });
}

export function probabilityData(
  forecast: Forecast,
  key: string,
  thresholds: number[],
) {
  const rows: number[][] = [];
  forecast.predictions.forEach((point, x) => {
    thresholds.forEach((threshold, y) => {
      const prediction = point.variables[key]?.binary?.find(
        (item) => item.threshold === threshold &&
          (item.operator === undefined || item.operator === "gte"),
      );
      if (prediction)
        rows.push([x, y, Math.round(prediction.probability * 100)]);
    });
  });
  return rows;
}
