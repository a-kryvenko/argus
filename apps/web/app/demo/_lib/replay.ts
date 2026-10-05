import type { Forecast } from '../../_utils/api';

export type DemoObservation = { time: string; values: Record<string, number | null> };
export type DemoBundle = {
  schema_version: 1;
  version: string;
  generated_at: string;
  event: { name: string; starts_at: string; source_url: string; description: string };
  observation_source: string;
  input_source: string;
  models: Record<string, { sha256: string; training_end: string; training_evidence: string }>;
  observations: DemoObservation[];
  releases: Record<string, Forecast[]>;
};

export function parseOffset(value: string | null): number {
  if (value === null || !/^-?\d+$/.test(value)) return -96;
  return Math.min(0, Math.max(-96, Number(value)));
}

export function replayTime(bundle: DemoBundle, offset: number): number {
  return Date.parse(bundle.event.starts_at) + offset * 3600000;
}

export function countdown(offset: number): string {
  return offset === 0 ? 'T0 · Event begins' : `T−${Math.abs(offset)} h · Until event`;
}

export function demoHref(path: string, offset: number): string {
  return `/demo${path === '/' ? '' : path}?offset=${offset}`;
}

export function selectRelease(bundle: DemoBundle, target: string, now: number): Forecast | null {
  // An absent hourly release is a gap, never a live or a future forecast.
  const issue = Math.floor(now / 3600000) * 3600000;
  return bundle.releases[target]?.find(release => Date.parse(release.issue_time) === issue) ?? null;
}

export function observationKnownAt(time: string, variable: string): number {
  const timestamp = Date.parse(time);
  const interval = ['kp', 'ap'].includes(variable) ? 3 * 3600000 : variable === 'f10_7' ? 24 * 3600000 : 3600000;
  return Math.floor(timestamp / interval) * interval + interval;
}

export function comparisonRows(forecast: Forecast | null, observations: DemoObservation[], variable: string,
  now: number, showFuture: boolean, horizon?: number | null) {
  const predictions = forecast?.predictions.filter(point => horizon == null || point.lead_hours <= horizon) ?? [];
  const end = predictions.length ? Math.max(...predictions.map(point => Date.parse(point.valid_time))) : now;
  const rows = new Map<number, { time: string; low: number | null; median: number | null; high: number | null;
    observed: number | null; actual: number | null }>();
  for (const point of observations) {
    const time = Date.parse(point.time);
    if (time < now - 24 * 3600000 || time > end) continue;
    rows.set(time, { time: point.time, low: null, median: null, high: null,
      observed: observationKnownAt(point.time, variable) <= now ? point.values[variable] ?? null : null,
      actual: observationKnownAt(point.time, variable) > now && showFuture ? point.values[variable] ?? null : null });
  }
  for (const point of predictions) {
    const time = Date.parse(point.valid_time);
    const values = point.variables[variable]?.continuous;
    rows.set(time, { time: point.valid_time, observed: null, actual: null, ...rows.get(time),
      low: values?.q10 ?? null, median: values?.q50 ?? null, high: values?.q90 ?? null });
  }
  // Keep the simulated clock in the plotted domain even when observations are missing.
  if (!rows.has(now)) rows.set(now, { time: new Date(now).toISOString(), low: null, median: null, high: null, observed: null, actual: null });
  return [...rows.entries()].sort(([a], [b]) => a - b).map(([, row]) => row);
}
