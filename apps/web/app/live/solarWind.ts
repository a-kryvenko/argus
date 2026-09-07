export type Metric = 'bx' | 'by' | 'bz' | 'bt' | 'v' | 'n' | 't';
export type Sample = {
  observed_at: string;
  received_at: string;
  value: number | null;
  spacecraft: string;
  quality: 'missing' | 'flagged' | 'unverified';
  provider_quality: number | null;
};
export type Metadata = {
  label: string;
  unit: string;
  coordinate_system: string | null;
  source_url: string;
  stale_after_seconds: number;
};
export type Latest = {
  generated_at: string;
  series: Record<Metric, Metadata & {
    latest: Sample | null;
    age_seconds: number | null;
    status: 'missing' | 'stale' | 'fresh';
  }>;
};
export type History = {
  from: string;
  to: string;
  series: Partial<Record<Metric, Metadata & { points: Sample[] }>>;
};
export type ChartPoint = { time: number } & Partial<Record<Metric, number | null>>;

// Null rows break lines across omitted minutes and spacecraft changes.
// Neither this presentation layer nor the API interpolates measurements.
export function chartPoints(history: History, metrics?: Metric[]): ChartPoint[] {
  const rows = new Map<number, ChartPoint>();
  const breaks = new Set<number>();
  for (const [key, series] of Object.entries(history.series)) {
    if (!series) continue;
    const metric = key as Metric;
    if (metrics && !metrics.includes(metric)) continue;
    let previous: Sample | undefined;
    for (const point of series.points) {
      const time = Date.parse(point.observed_at);
      if (previous) {
        const before = Date.parse(previous.observed_at);
        if (time - before > 90_000 || point.spacecraft !== previous.spacecraft) {
          breaks.add(before + Math.min(60_000, (time - before) / 2));
        }
      }
      const row = rows.get(time) ?? { time };
      row[metric] = point.quality === 'flagged' ? null : point.value;
      rows.set(time, row);
      previous = point;
    }
  }
  for (const time of breaks) {
    if (!rows.has(time)) rows.set(time, { time });
  }
  return [...rows.values()].sort((a, b) => a.time - b.time);
}
