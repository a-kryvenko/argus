'use client';
import type { Derived, LiveSnapshot } from './useLiveQuery';
import styles from './page.module.css';

const number = (value: number | null | undefined) => value == null ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: 2 });
const reasons: Record<string, string> = { stale: 'delayed measurements', source_changed: 'spacecraft changed', insufficient_coverage: 'insufficient coverage', missing_latest: 'latest measurement unavailable' };
function trend(point: Derived | undefined) {
  if (!point || point.value == null) return `unavailable (${reasons[point?.reason ?? ''] ?? 'insufficient data'})`;
  return `${point.value > 0 ? '+' : ''}${number(point.value)} ${point.unit ?? ''}`;
}

export default function ObservationSummary({ snapshot }: { snapshot: LiveSnapshot }) {
  const { data: summary, failed, receivedAt: lastSuccess, now: clock } = snapshot;
  const oldSnapshot = clock && lastSuccess ? clock-lastSuccess > 120000 : false;
  return <section className={styles.summary} aria-label="Observation summary">
    <h2>At a glance</h2>
    {failed && <p role="alert" className={styles.warning}>Summary could not be refreshed. {summary ? 'Showing the last snapshot.' : 'Retrying in one minute.'}</p>}
    {oldSnapshot && <p className={styles.warning}>This snapshot is more than two minutes old.</p>}
    {!summary && !failed && <p role="status">Loading observation summary…</p>}
    {summary && <>
      <div className={styles.summaryGrid}>
        {(['v','bz'] as const).map(metric => {
          const series = summary.solar_wind[metric];
          if (metric === 'v') {
            return <div key={metric}><h3>Solar wind speed</h3><p>{number(series.latest?.value)} {series.unit}{series.status !== 'fresh' ? ` · ${series.status}` : ''}{series.latest?.quality === 'flagged' ? ' · flagged' : ''}</p><small>1h change: {trend(summary.changes_1h[metric])}</small></div>;
          }
          const point = series.latest;
          const elapsed = clock && lastSuccess ? Math.max(0, (clock-lastSuccess)/1000) : 0;
          const age = series.age_seconds == null ? null : series.age_seconds + elapsed;
          const fresh = series.status === 'fresh' && age != null && age <= series.stale_after_seconds;
          const usable = fresh && point?.value != null && point.quality !== 'flagged';
          const duration = summary.southward_bz;
          const showDuration = usable && (point?.value ?? 0) < 0 && duration.status !== 'unavailable'
            && duration.value != null && duration.value > 0 && duration.as_of === point?.observed_at;
          const change = summary.changes_1h.bz;
          const showChange = usable && change?.status === 'available' && change.value != null;
          return <div key={metric}>
            <h3>Bz</h3>
            <p>{number(point?.value)} {series.unit}{point?.quality === 'flagged' ? ' · flagged' : ''}</p>
            {showDuration && <small className={styles.summaryLine}>Negative for {duration.status === 'lower_bound' ? 'at least ' : ''}{number(duration.value)} min through {new Date(duration.as_of ?? summary.generated_at).toISOString().slice(11,16)} UTC</small>}
            {showChange && <small className={styles.summaryLine}>1h change: {trend(change)}</small>}
            <small className={styles.summaryLine}>{point?.value == null ? 'Measurement unavailable' : fresh ? 'Recent' : 'Delayed'}{age != null && ` · ${Math.floor(age/60)} min old`}</small>
            <details className={styles.summaryDetails}>
              <summary>Details</summary>
              {point && <small className={styles.summaryLine}>Latest Bz: {new Date(point.observed_at).toISOString().replace('T', ' ').slice(0,16)} UTC</small>}
              {!showDuration && (point?.value == null || point.value < 0) && <small className={styles.summaryLine}>Negative duration: {reasons[!fresh && point?.value != null ? 'stale' : duration.reason ?? ''] ?? 'insufficient data'}</small>}
              {!showChange && <small className={styles.summaryLine}>1h change: {fresh ? trend(change) : 'unavailable (delayed measurements)'}</small>}
            </details>
          </div>;
        })}
        {(['kp','dst'] as const).map(metric => {
          const series = summary.geomagnetic[metric];
          return <div key={metric}><h3>{metric === 'kp' ? 'Kp' : series.label}</h3><p>{number(series.latest?.value)} {series.unit}{series.status !== 'fresh' ? ` · ${series.status}` : ''}{series.latest?.quality === 'flagged' ? ' · flagged' : ''}</p><small>{series.latest ? `${new Date(series.latest.interval_start).toISOString().slice(5,16).replace('T',' ')} – ${new Date(series.latest.interval_end).toISOString().slice(11,16)} UTC` : 'No stored interval'}</small></div>;
        })}
      </div>
      <p className={styles.sampleTime}>Snapshot: {new Date(summary.generated_at).toISOString().replace('T',' ').slice(0,19)} UTC. Changes compare five-minute means one hour apart and require ≥80% coverage in both windows and the last hour.</p>
      <a href="/api/v1/public/observations/summary">Summary · JSON</a>
    </>}
  </section>;
}
