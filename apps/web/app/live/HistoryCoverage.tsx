import styles from './page.module.css';

export type Coverage = {
  expected_slots: number; usable_slots: number; missing_slots: number; invalid_slots: number;
  percent: number | null; resolution_seconds: number;
  gaps: { from: string; to: string; reason: 'missing' | 'invalid' | 'partial'; slots: number }[];
};
export type Processing = { available_buckets: number; expected_buckets: number; unavailable_buckets: number; recalculation_pending_buckets: number };
const stamp = (value: string) => new Date(value).toISOString().replace('T', ' ').slice(0, 16);
export default function HistoryCoverage({ coverage, label, processing }: { coverage?: Coverage; label: string; processing?: Processing }) {
  if (!coverage) return null;
  return <div className={styles.description}>
    <p>{label} coverage{processing ? ' in calculated windows' : ''}: {coverage.percent == null ? processing ? 'Not yet known' : 'No expected samples in this period' : `${coverage.percent}% · ${coverage.usable_slots} of ${coverage.expected_slots} usable ${coverage.resolution_seconds === 60 ? 'minute samples' : 'intervals'}`}</p>
    {processing && processing.unavailable_buckets > 0 && <p>Calculated windows: {processing.available_buckets} of {processing.expected_buckets}. Coverage outside these windows is unknown.</p>}
    {processing && processing.recalculation_pending_buckets > 0 && <p>{processing.recalculation_pending_buckets} windows await recalculation. Showing their last calculated values and coverage.</p>}
    {coverage.gaps.length > 0 && <details className={styles.additional}>
      <summary>{coverage.missing_slots} missing · {coverage.invalid_slots} unusable · gap details</summary>
      <p>Missing records and measurements with missing values or provider quality flags are excluded. Recent gaps may reflect publication delay.</p>
      <ul>{coverage.gaps.slice(0, 50).map(gap => <li key={gap.from}>{stamp(gap.from)} – {stamp(gap.to)} UTC · {gap.slots} {gap.reason === 'missing' ? 'missing' : gap.reason === 'partial' ? 'missing or unusable' : 'unusable'}</li>)}</ul>
      {coverage.gaps.length > 50 && <p>Showing the first 50 of {coverage.gaps.length} gaps. The history JSON contains the full list.</p>}
    </details>}
  </div>;
}
