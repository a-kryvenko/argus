export type Attention = 'not_assessed' | 'within_reference' | 'watch' | 'review' | 'high';
export function utc(value: string) {
  return new Date(value).toLocaleString('en-GB', { timeZone: 'UTC', hour12: false });
}
export type Status = 'current' | 'partial' | 'unavailable' | 'stale' | 'trajectory_expired' | 'profile_changed' | 'paused';
export type TrajectoryPoint = { time: string; altitude_km: number; latitude_deg: number; longitude_deg: number; reference_density_kg_m3?: number | null };
export type Trajectory = { source: string; generated_at: string; valid_until: string; coordinates: 'geocentric_spherical'; points: TrajectoryPoint[] };
export type Reference = { label: string; source: string; prepared_at: string; kind?: 'operator_reference' | 'model_geomagnetic_reference'; spatial_support?: 'longitude_mean'; method?: string | null };
export type Spacecraft = { apoapsis_km: number; periapsis_km: number; inclination_deg: number; weight_kg: number | null; effective_area_m2: number | null };
export type DensityProfileInput = Spacecraft & { name: string; policy: { review_ratio: number; high_ratio: number } };
export type DensityProfile = Omit<DensityProfileInput, keyof Spacecraft> & { [K in keyof Spacecraft]: Spacecraft[K] | null } & { enabled: boolean; trajectory?: Trajectory | null; reference?: Reference | null; id: string; revision: number; updated_at: string };
export type Reason = { code: string; message: string; source_id: string | null };
export type Sample = Omit<TrajectoryPoint, 'longitude_deg'> & { longitude_deg: number | null } & { density_kg_m3: number | null; ratio: number | null; attention: Attention; reasons: Reason[] };
export type Interval = { start: string; end: string; attention: Attention; peak_ratio: number | null; peak_density_kg_m3: number | null; sample_count: number; reasons: Reason[]; sampled_duration_minutes?: number; peak_location?: { time: string; altitude_km: number; latitude_deg: number; longitude_deg: number | null } | null };
export type DensityRun = { id: string; profile_id: string; profile_revision: number; profile_snapshot: Omit<DensityProfile, 'id' | 'revision' | 'updated_at'>; assessment: {
  computed_at: string; attention: Attention; data_status: Status; samples: Sample[]; intervals: Interval[]; limitations: string[];
  source: { assessment_basis?: 'supplied_trajectory' | 'orbit_envelope'; density_release_id: string | null; density_issue_time: string | null; density_observed_at: string | null; density_model: string | null;
    cme_release_id: string | null; cme_issue_time: string | null; cme_status: 'current' | 'stale' | 'unavailable';
    trajectory_source: string | null; trajectory_generated_at: string | null; reference: Reference | null };
} };
export type DensityRunSummary = { id: string; profile_revision: number; computed_at: string };
export type DensityItem = { profile: Pick<DensityProfile, 'id' | 'name' | 'revision' | 'updated_at' | 'enabled'> & Partial<Pick<DensityProfile, 'apoapsis_km' | 'periapsis_km' | 'inclination_deg'>>; latest: DensityRunSummary | null; attention: Attention; data_status: Status; cme_status: 'current' | 'stale' | 'unavailable'; details?: string[] };

export function spacecraftStatus(item: DensityItem) {
  const problems: string[] = [];
  if (!item.profile.enabled || item.data_status === 'paused') return { warning: true, details: ['Monitoring is paused.'] };
  const dataProblems: Partial<Record<Status, string>> = {
    partial: 'Density or reference coverage is incomplete.', unavailable: 'Density forecast is unavailable.',
    stale: 'Density forecast is out of date.', trajectory_expired: 'Saved trajectory has expired. Update the spacecraft.',
    profile_changed: 'Spacecraft parameters have changed. A new assessment is needed.',
  };
  if (item.data_status !== 'current') problems.push(dataProblems[item.data_status]!);
  if (item.attention === 'not_assessed' && item.data_status === 'current') problems.push('Density risk has not been assessed.');
  if (item.details?.length) problems.push(...item.details);
  else if (item.attention === 'review' || item.attention === 'high') problems.push(item.attention === 'high' ? 'Density exceeds the high warning threshold.' : 'Density exceeds the review threshold.');
  else if (item.attention === 'watch') problems.push('A predicted CME arrival falls within the forecast period.');
  if (item.cme_status !== 'current') problems.push(item.cme_status === 'stale' ? 'CME forecast is out of date.' : 'CME forecast is unavailable.');
  if ([item.profile.apoapsis_km, item.profile.periapsis_km, item.profile.inclination_deg].some(value => value == null)) problems.push('Orbital parameters are incomplete.');
  return { warning: problems.length > 0, details: [...new Set(problems)] };
}
export type DensityEvent = { id: string; profile_id: string; run_id: string; detected_at: string; kind: string; attention: Attention; message: string; acknowledged_at: string | null };
export type DensityEventPage = { items: DensityEvent[]; total: number };
export const attentionLabel: Record<Attention, string> = { not_assessed: 'Not assessed', within_reference: 'Below configured ratios', watch: 'CME watch', review: 'Review density increase', high: 'High density ratio' };
export const statusLabel: Record<Status, string> = { current: 'Current density coverage', partial: 'Incomplete anomaly coverage', unavailable: 'Density unavailable', stale: 'Stale density', trajectory_expired: 'Trajectory expired', profile_changed: 'Spacecraft profile changed', paused: 'Paused' };
