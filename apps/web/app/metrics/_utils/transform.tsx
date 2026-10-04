import type { BinaryMetricsPoint, ForecastMetrics } from '../../_utils/api';

export type VariableMetrics = ForecastMetrics['variables'][string];
export type ScoreKey = Exclude<keyof BinaryMetricsPoint, 'lead_hours' | 'reliability'>;
export type MetricRow = { lead_hours: number; values: Record<string, number | null> };
export const scoreLabels: Record<ScoreKey, string> = {
  brier_score: 'Brier score', roc_auc: 'ROC AUC', average_precision: 'Average precision',
  threat_score: 'Threat score', heidke_skill_score: 'Heidke skill score',
};
export const scoreNotes: Record<ScoreKey, string> = {
  brier_score: 'Mean squared error of event probabilities · lower is better.',
  roc_auc: 'Area under the ROC curve · higher is better.',
  average_precision: 'Summary of precision and recall · higher is better.',
  threat_score: 'Hits / (hits + misses + false alarms) · higher is better.',
  heidke_skill_score: 'Skill relative to chance agreement · 0 means no skill over chance; 1 is perfect.',
};
export const metricLabel = (key: string) => ({ mae: 'MAE', rmse: 'RMSE', bias: 'Bias', coverage_80: '80% interval coverage', interval_width_80: '80% interval width', n: 'Samples' }[key] ?? key.replaceAll('_', ' '));
const integerFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
const fractionFormat = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 });
const exactFormat = new Intl.NumberFormat('en-US', { useGrouping: false, maximumSignificantDigits: 21 });
export const metricNumber = (value: number | null | undefined) => {
  if (value == null || !Number.isFinite(value)) return '—';
  const magnitude = Math.abs(value);
  if (magnitude > 0 && magnitude < 0.001) return value > 0 ? '<0.001' : '>-0.001';
  return (magnitude >= 1 ? integerFormat : fractionFormat).format(value);
};
export const metricTitle = (value: number | null | undefined) => value == null || !Number.isFinite(value)
  ? undefined : exactFormat.format(value);
export function metricUnit(key: string, unit: string) {
  if (['mae', 'rmse', 'bias', 'interval_width_80'].includes(key) || /^q\d+_pinball$/.test(key)) return unit;
  if (key.startsWith('coverage') || ['lower_tail', 'upper_tail'].includes(key)) return 'fraction';
  if (key === 'n' || key === 'scheduled') return 'samples';
  return '';
}
export function leadHours(variable?: VariableMetrics) {
  return [...new Set([...(variable?.continuous?.by_lead_hour.map(row => row.lead_hours) ?? []),
    ...(variable?.binary.flatMap(series => series.by_lead_hour.map(row => row.lead_hours)) ?? [])])].sort((a, b) => a - b);
}
export function continuousKeys(variable?: VariableMetrics) {
  return [...new Set(variable?.continuous?.by_lead_hour.flatMap(row => Object.keys(row.values)) ?? [])];
}
export function binaryRows(variable: VariableMetrics, score: ScoreKey): MetricRow[] {
  const hours = [...new Set(variable.binary.flatMap(series => series.by_lead_hour.map(row => row.lead_hours)))].sort((a, b) => a - b);
  return hours.map(lead_hours => ({ lead_hours, values: Object.fromEntries(variable.binary.map(series => [String(series.threshold), series.by_lead_hour.find(row => row.lead_hours === lead_hours)?.[score] ?? null])) }));
}
// A null separator prevents drawing a line across omitted hourly evaluations.
export function chartRows(rows: MetricRow[]): MetricRow[] {
  const sorted = [...rows].sort((a, b) => a.lead_hours - b.lead_hours);
  return sorted.flatMap((row, index) => index > 0 && row.lead_hours - sorted[index - 1].lead_hours > 1
    ? [{ lead_hours: sorted[index - 1].lead_hours + 1, values: {} }, row] : [row]);
}
