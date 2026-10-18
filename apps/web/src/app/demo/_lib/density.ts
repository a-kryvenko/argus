export type DensityVariant = 'jb2008_fixed' | 'jb2008_forecast' | 'dtm2020_fixed' | 'dtm2020_forecast';
export type Attention = 'within_reference' | 'review' | 'high' | 'not_assessed';
export type DensityPoint = { time: string; density_kg_m3: number | null; ratio: number | null; attention: Attention };
export type DensityInterval = { start: string; end: string; attention: Attention; sample_count: number; peak_ratio: number };
export type DensityRelease = { issue_time: string; attention: Attention; intervals: DensityInterval[]; removed_warning_samples: number; predictions: DensityPoint[] };
export type DensityJournal = { variant: DensityVariant; issue_time: string; attention: Attention; warning_intervals: number; usable_samples: number; missing_samples: number; overlapping_samples: number; changed_samples: number; removed_warning_samples: number; appeared_warning_samples: number };
export type DensityScore = { variant: DensityVariant; threshold_ratio: number; lead_hours: number; requested: number; usable: number; excluded: number; true_positive: number; false_positive: number; false_negative: number; true_negative: number; precision: number | null; recall: number | null };
export type DensityScenario = {
  schema_version: 1; status: string; spacecraft: string;
  reference: { method: string; density_kg_m3: number; start: string; end_exclusive: string; samples: number; expected_samples: number; altitude_range_km: number[]; availability: string };
  policy: { review_ratio: number; high_ratio: number };
  variants: DensityVariant[]; releases: Record<DensityVariant, DensityRelease[]>;
  observations: Array<{ time: string; density_kg_m3: number | null; attention: Attention }>;
  density_metrics: Array<{ variant: DensityVariant; lead_hours: number; n: number; mape_pct: number | null; bias_pct: number | null; underestimate_gt_30pct: number | null }>;
  scores: DensityScore[]; journal: DensityJournal[];
  events: Array<{ variant: DensityVariant; threshold_ratio: number; onset_first_warning_issue: string | null; onset_notice_hours: number | null; start: string; end: string; sample_count: number; first_warning_issue: string | null; notice_hours: number | null; warned_before_start: boolean; missed: boolean }>;
  limitations: string[];
};

export function densityRelease(scenario: DensityScenario, variant: DensityVariant, now: number) {
  const issue = Math.floor(now / 3600000) * 3600000;
  return scenario.releases[variant]?.find(release => Date.parse(release.issue_time) === issue) ?? null;
}

export function densityRows(scenario: DensityScenario, release: DensityRelease | null, now: number, showFuture: boolean) {
  const end = release?.predictions.at(-1)?.time;
  const last = end ? Date.parse(end) : now;
  const rows = new Map<number, { timestamp: number; forecast: number | null; observed: number | null; actual: number | null }>();
  for (const point of scenario.observations) {
    const time = Date.parse(point.time);
    if (time < now - 24 * 3600000 || time > last) continue;
    const value = point.density_kg_m3 == null ? null : point.density_kg_m3 / scenario.reference.density_kg_m3;
    rows.set(time, { timestamp: time, forecast: null, observed: time <= now ? value : null, actual: time > now && showFuture ? value : null });
  }
  for (const point of release?.predictions ?? []) {
    const time = Date.parse(point.time);
    rows.set(time, { timestamp: time, observed: null, actual: null, ...rows.get(time), forecast: point.ratio });
  }
  return [...rows.values()].sort((a, b) => a.timestamp - b.timestamp);
}

export function visibleDensityJournal(scenario: DensityScenario, variant: DensityVariant, now: number) {
  return scenario.journal.filter(entry => entry.variant === variant && Date.parse(entry.issue_time) <= now).slice(-12).reverse();
}
