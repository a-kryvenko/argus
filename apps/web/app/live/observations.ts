import { windUnits, windCoordinates, type Metric } from './solarWind';
import { indexDefinitions, type IndexMetric } from './geomagnetic';
import type { LiveSnapshot } from './useLiveQuery';

export type Selection = Metric | IndexMetric;
export const windLabels: Record<Metric, string> = {
  v: 'Solar wind speed', n: 'Proton density', bz: 'Bz', bt: 'Total field Bt',
  bx: 'Bx', by: 'By', t: 'Proton temperature',
};
export const metricColors: Record<Selection, string> = {
  v: '#65b5eb', n: '#b1a0df', bz: '#65c9b3', bt: '#d2b984', bx: '#82aec9', by: '#a7bfd1',
  t: '#cba88b', kp: '#8dace0', dst: '#73b9bc',
};
export const selections: Selection[] = ['v', 'n', 'bz', 'bt', 'kp', 'dst', 'bx', 'by', 't'];
export function isWind(metric: Selection): metric is Metric { return metric !== 'kp' && metric !== 'dst'; }
export function definition(metric: Selection) {
  return isWind(metric)
    ? { label: windLabels[metric], unit: windUnits[metric], frame: windCoordinates[metric] }
    : { ...indexDefinitions[metric], frame: undefined };
}
export function number(value: number | null | undefined, digits = 1) {
  return value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: digits });
}
export function stamp(value: string | number | undefined) {
  return value === undefined ? '—' : new Date(value).toISOString().replace('T', ' ').slice(0, 16);
}

// A successful refresh does not make an old or provider-flagged sample usable.
export function measurement(metric: Selection, snapshot: LiveSnapshot) {
  const series = isWind(metric) ? snapshot.data?.solar_wind[metric] : snapshot.data?.geomagnetic[metric];
  const point = series?.latest;
  const elapsed = snapshot.now && snapshot.receivedAt ? Math.max(0, (snapshot.now - snapshot.receivedAt) / 1000) : 0;
  let age: number | null = null;
  if (series && 'age_seconds' in series && series.age_seconds != null) age = series.age_seconds + elapsed;
  if (point && 'interval_end' in point && point.interval_end && snapshot.data) {
    age = Math.max(0, (Date.parse(snapshot.data.generated_at) - Date.parse(point.interval_end)) / 1000 + elapsed);
  }
  const stale = series?.status === 'stale' || (age != null && series != null && age > series.stale_after_seconds);
  const outdated = snapshot.now && snapshot.receivedAt ? snapshot.now - snapshot.receivedAt > 120000 : false;
  const flagged = point?.quality === 'flagged';
  const missing = point?.value == null || point.quality === 'missing' || series?.status === 'missing';
  const usable = !missing && !flagged && !stale && !outdated && series?.status === 'fresh';
  const status = !snapshot.data ? snapshot.failed ? 'Unavailable' : 'Loading' : missing ? 'No data' : flagged ? 'Flagged' :
    stale ? 'Delayed' : outdated ? 'Old snapshot' : 'Recent';
  return { series, point, age, stale, outdated, flagged, missing, usable, status };
}
