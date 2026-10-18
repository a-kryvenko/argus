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
  const pending = processing?.recalculation_pending_buckets ?? 0;
  const unavailable = processing?.unavailable_buckets ?? 0;
  return <details className={styles.coverage}>
    <summary><span>{label} <span className={styles.mono}>{coverage.percent == null ? '—' : `${coverage.percent}%`}</span> coverage</span>
      <span className={coverage.invalid_slots || coverage.missing_slots || pending || unavailable ? styles.warning : styles.muted}>
        {[unavailable ? `${unavailable} windows unavailable` : '', pending ? `${pending} pending` : '',
          coverage.missing_slots || coverage.invalid_slots ? `${coverage.missing_slots} missing · ${coverage.invalid_slots} unusable` : ''].filter(Boolean).join(' · ')
          || (coverage.percent == null ? 'No evaluated samples' : 'Complete')}
      </span>
    </summary>
    <div className={styles.coverageDetails}>
      <p>{coverage.percent == null ? processing ? 'Coverage is not yet known.' : 'No expected samples in this period.' : `${coverage.usable_slots} of ${coverage.expected_slots} usable ${coverage.resolution_seconds === 60 ? 'minute samples' : 'intervals'}.`}</p>
      {processing && <p>Coverage describes calculated windows only. Calculated: {processing.available_buckets} of {processing.expected_buckets}. Coverage outside these windows is unknown.</p>}
      {pending > 0 && <p>{pending} windows await recalculation. Showing their last calculated values and coverage.</p>}
      {coverage.gaps.length > 0 && <>
        <p>Missing values and provider flags are excluded. Recent gaps may reflect publication delay. Aggregate gaps identify affected windows, not exact missing-minute times.</p>
        <ul>{coverage.gaps.slice(0, 50).map(gap => <li key={gap.from}>{stamp(gap.from)} – {stamp(gap.to)} UTC · {gap.slots} {gap.reason === 'missing' ? 'missing' : gap.reason === 'partial' ? 'missing or unusable' : 'unusable'}</li>)}</ul>
        {coverage.gaps.length > 50 && <p>Showing the first 50 of {coverage.gaps.length} gaps. The history JSON contains the full list.</p>}
      </>}
    </div>
  </details>;
}
