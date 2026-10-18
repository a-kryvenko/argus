'use client';
import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react';
import type { Derived, LiveSnapshot } from './useLiveQuery';
import { definition, measurement, metricColors, number, type Selection } from './observations';
import styles from './page.module.css';

const metrics: Selection[] = ['v', 'n', 'bz', 'bt', 'kp', 'dst'];
const reasons: Record<string, string> = { stale: 'Delayed measurements', source_changed: 'Spacecraft changed', insufficient_coverage: 'Insufficient coverage', missing_latest: 'Latest measurement unavailable' };
function trend(point: Derived | undefined, unit: string) {
  if (!point || point.value == null || point.status === 'unavailable') return reasons[point?.reason ?? ''] ?? 'Change unavailable';
  return `1h change: ${point.value > 0 ? '+' : ''}${number(point.value, 2)} ${unit}`;
}

export default function ObservationSummary({ snapshot, selected, onSelect }: {
  snapshot: LiveSnapshot; selected: Selection; onSelect: (metric: Selection) => void;
}) {
  const bz = measurement('bz', snapshot);
  const duration = snapshot.data?.southward_bz;
  const bzPoint = snapshot.data?.solar_wind.bz?.latest;
  const showDuration = bz.usable && (bzPoint?.value ?? 0) < 0 && duration?.status !== 'unavailable'
    && duration?.value != null && duration.value > 0 && duration.as_of === bzPoint?.observed_at;
  return <section className={styles.summary} aria-label="Observation summary">
    {snapshot.failed && <p role="alert" className={styles.notice}>Summary could not be refreshed. {snapshot.data ? 'Showing the last snapshot.' : 'Retrying in one minute.'}</p>}
    {bz.outdated && <p className={styles.notice}>This snapshot is more than two minutes old.</p>}
    <div className={styles.summaryGrid}>
      {metrics.map(metric => {
        const { label, unit } = definition(metric);
        const { point, usable, status } = measurement(metric, snapshot);
        const change = snapshot.data?.changes_1h[metric];
        const TrendIcon = change?.value == null || !usable ? Minus : change.value < 0 ? ArrowDownRight : ArrowUpRight;
        return <button key={metric} className={styles.metricCard} aria-pressed={selected === metric}
          aria-label={`Inspect ${label}`} onClick={() => onSelect(metric)} style={{ '--metric-color': metricColors[metric] } as React.CSSProperties}>
          <span className={styles.metricLabel}><span>{label}</span><span className={styles.metricCode}>{metric.toUpperCase()}</span></span>
          <span className={styles.metricValue}>{number(point?.value, metric === 'kp' ? 2 : 1)} <small>{unit}</small></span>
          <span className={styles.metricState} data-warning={snapshot.data && !usable}><span className={styles.dot} />{status}</span>
          <span className={styles.metricTrend}>
            {metric === 'kp' || metric === 'dst' ? <>{metric === 'kp' ? '3-hour' : 'Hourly'} interval · provisional</> :
              <><TrendIcon size={13} aria-hidden="true" />{usable ? trend(change, unit) : 'Change unavailable'}</>}
          </span>
        </button>;
      })}
    </div>
    <div className={styles.contextStrip}>
      <span className={styles.contextLabel}><span className={styles.dot} />SOUTHWARD Bz</span>
      <span>{showDuration ? <>Negative for {duration.status === 'lower_bound' ? 'at least ' : ''}{number(duration.value)} min <span className={styles.muted}>through {duration.as_of?.slice(11, 16)} UTC</span></> :
        !snapshot.data ? 'Waiting for observations' : !bz.usable ? 'Duration unavailable · check measurement quality and age' :
          (bzPoint?.value ?? 0) >= 0 ? 'Latest Bz is not southward' : reasons[duration?.reason ?? ''] ?? 'Duration unavailable'}</span>
      <span className={styles.contextHint}>Observed conditions · no impact score</span>
    </div>
  </section>;
}
