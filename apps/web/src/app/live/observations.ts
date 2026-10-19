import { windUnits, windCoordinates, type Metric } from './solarWind';
import { indexDefinitions, type IndexMetric } from './geomagnetic';
import type { LiveSnapshot } from './useLiveQuery';

export type Selection = Metric | IndexMetric | 'f10_7';
export const windLabels: Record<Metric, string> = {
  v: 'Solar wind speed', n: 'Proton density', bz: 'Bz', bt: 'Total field Bt',
  bx: 'Bx', by: 'By', t: 'Proton temperature',
};
export const metricColors: Record<Selection, string> = {
  v: '#65b5eb', n: '#b1a0df', bz: '#65c9b3', bt: '#d2b984', bx: '#82aec9', by: '#a7bfd1',
  t: '#cba88b', kp: '#8dace0', dst: '#73b9bc', f10_7: '#d2b984',
};
export const selections: Selection[] = ['v', 'n', 'bz', 'bt', 'kp', 'dst', 'bx', 'by', 't', 'f10_7'];
export function isWind(metric: Selection): metric is Metric { return metric !== 'kp' && metric !== 'dst' && metric !== 'f10_7'; }
export function definition(metric: Selection) {
  if (metric === 'f10_7') return { label: 'F10.7', unit: 'sfu', frame: undefined };
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
  const point = snapshot.data?.[metric];
  const staleAfter = metric === 'f10_7' ? 172800 : isWind(metric) ? 600 : metric === 'kp' ? 14400 : 7200;
  const series = point ? { stale_after_seconds: staleAfter } : undefined;
  const now = snapshot.now ?? snapshot.receivedAt;
  const measuredAt = point && ('observed_at' in point ? point.observed_at : point.interval_end);
  const age = measuredAt && now != null ? Math.max(0, (now - Date.parse(measuredAt)) / 1000) : null;
  const stale = age != null && age > staleAfter;
  const outdated = snapshot.now && snapshot.receivedAt ? snapshot.now - snapshot.receivedAt > 120000 : false;
  const flagged = point?.quality === 'flagged';
  const missing = point?.value == null || point.quality === 'missing';
  const usable = !missing && !flagged && !stale && !outdated;
  const status = !snapshot.data ? snapshot.failed ? 'Unavailable' : 'Loading' : missing ? 'No data' : flagged ? 'Flagged' :
    stale ? 'Delayed' : outdated ? 'Old snapshot' : 'Recent';
  return { series, point, age, stale, outdated, flagged, missing, usable, status };
}
