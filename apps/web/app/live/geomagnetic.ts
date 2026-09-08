import type { Coverage } from './HistoryCoverage';
export type IndexMetric = 'kp' | 'dst';
export type IndexSample = {
  interval_start: string;
  interval_end: string;
  interval_status: 'in_progress' | 'completed';
  value: number | null;
  quality: 'missing' | 'flagged' | 'unverified';
  received_at: string;
  station_count: number | null;
};
export type IndexMetadata = {
  label: string;
  unit: string;
  source: string;
  data_status: 'estimated' | 'realtime';
  stale_after_seconds: number;
};
export type IndexLatest = {
  generated_at: string;
  series: Record<IndexMetric, IndexMetadata & {
    latest: IndexSample | null;
    lag_seconds: number | null;
    status: 'fresh' | 'stale' | 'missing';
  }>;
};
export type IndexHistory = {
  from: string;
  to: string;
  series: Record<IndexMetric, IndexMetadata & { points: IndexSample[]; coverage?: Coverage }>;
};

export function intervalBounds(point: IndexSample, from: number, to: number): [number, number] | null {
  if (point.value == null || point.quality === 'flagged') return null;
  const left = Math.max(from, Date.parse(point.interval_start));
  const right = Math.min(to, Date.parse(point.interval_end));
  return left < right ? [left, right] : null;
}
