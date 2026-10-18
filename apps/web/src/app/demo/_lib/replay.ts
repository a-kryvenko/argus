import type { Forecast } from '@/forecast/types';
import type { DensityScenario } from './density';

export type DemoObservation = { time: string; values: Record<string, number | null> };
export type DemoEventReport = {
  issue_time: string;
  events: Array<{ event_id: string; status: string; available_at: string; arrival_at: string | null }>;
  cme_speed?: Forecast['cme_speed'];
};
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
  density_scenario?: DensityScenario;
  density_scenario_available?: boolean;
  event_releases?: DemoEventReport[];
};

export type DemoPageBundle = DemoBundle & {
  available_variables_by_product: Record<string, string[]>;
  density_scenario_available: boolean;
};

export function demoPageBundle(bundle: DemoBundle, product?: string, includeDensity = false): DemoPageBundle {
  const { density_scenario, ...shared } = bundle;
  return {
    ...shared,
    available_variables_by_product: Object.fromEntries(Object.entries(bundle.releases)
      .filter(([, releases]) => releases.length)
      .map(([target, releases]) => [target, [...new Set(releases.flatMap(release => release.available_variables))]])),
    releases: product && bundle.releases[product] ? { [product]: bundle.releases[product] } : {},
    density_scenario_available: !!density_scenario,
    ...(includeDensity ? { density_scenario } : {}),
  };
}

export function parseOffset(value: string | null): number {
  if (value === null || !/^-?\d+$/.test(value)) return -96;
  return Math.min(72, Math.max(-96, Number(value)));
}

export function replayTime(bundle: DemoBundle, offset: number): number {
  return Date.parse(bundle.event.starts_at) + offset * 3600000;
}

export function countdown(offset: number): string {
  return offset === 0 ? 'T0 · Event begins' : offset < 0 ? `T−${Math.abs(offset)} h · Until event` : `T+${offset} h · Since event`;
}

export function demoHref(path: string, offset: number): string {
  return `/demo${path === '/' ? '' : path}?offset=${offset}`;
}

export function selectRelease(bundle: DemoBundle, target: string, now: number): Forecast | null {
  // An absent hourly release is a gap, never a live or a future forecast.
  const issue = Math.floor(now / 3600000) * 3600000;
  return bundle.releases[target]?.find(release => Date.parse(release.issue_time) === issue) ?? null;
}

export function selectEventRelease(bundle: DemoBundle, now: number): DemoEventReport | null {
  const issue = Math.floor(now / 3600000) * 3600000;
  return bundle.event_releases?.find(release => Date.parse(release.issue_time) === issue) ?? null;
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
    observed: number | null; actual: number | null; cmeNominal?: number; cmeEarly?: number; cmeLate?: number;
    cmeRange?: [number, number] }>();
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
  if (variable === 'v' && forecast?.cme_speed?.events.length && forecast.cme_speed.state !== 'unavailable') {
    for (const point of forecast.cme_speed.points) {
      const row = rows.get(Date.parse(point.valid_time));
      if (row) Object.assign(row, { cmeNominal: point.speed_nominal, cmeEarly: point.speed_early,
        cmeLate: point.speed_late, cmeRange: [point.scenario_lower, point.scenario_upper] });
    }
  }
  // Keep the simulated clock in the plotted domain even when observations are missing.
  if (!rows.has(now)) rows.set(now, { time: new Date(now).toISOString(), low: null, median: null, high: null, observed: null, actual: null });
  return [...rows.entries()].sort(([a], [b]) => a - b).map(([, row]) => row);
}
