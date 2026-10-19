'use client';
import { useState } from 'react';
import ObservationInspector from './ObservationInspector';
import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react';
import type { LiveSnapshot } from './useLiveQuery';
import { definition, measurement, number, type Selection } from './observations';
import styles from './page.module.css';

const metrics: Selection[] = ['v', 'n', 'bz', 'f10_7', 'kp', 'dst'];
function trend(change: number | null | undefined, unit: string) {
  if (change == null) return 'Change unavailable';
  return `1h change: ${change > 0 ? '+' : ''}${number(change, 2)} ${unit}`;
}

export default function ObservationSummary({ snapshot }: { snapshot: LiveSnapshot }) {
  const [selected, setSelected] = useState<Selection>('v');
  const bz = measurement('bz', snapshot);
  return <div className={styles.currentObservations}><section className={styles.summary} aria-label="Observation summary">
    {snapshot.failed && <p role="alert" className={styles.notice}>Summary could not be refreshed. {snapshot.data ? 'Showing the last snapshot.' : 'Retrying in one minute.'}</p>}
    {bz.outdated && <p className={styles.notice}>This snapshot is more than two minutes old.</p>}
    <div className={styles.summaryGrid}>
      {metrics.map(metric => {
        const { label: metricLabel, unit } = definition(metric);
        const label = metric === 'bz' ? 'Magnetic field · Bz / Bt' : metricLabel;
        const { point, usable, status } = measurement(metric, snapshot);
        const bt = metric === 'bz' ? measurement('bt', snapshot) : undefined;
        const change = snapshot.data?.changes_1h?.[metric];
        const TrendIcon = change == null || !usable ? Minus : change < 0 ? ArrowDownRight : ArrowUpRight;
        return <button key={metric} className={styles.metricCard} aria-pressed={selected === metric}
          aria-label={`Inspect ${label}`} title={`View ${label} details`} aria-controls="observation-inspector" onClick={() => setSelected(metric)}>
          {!bt && <span className={styles.metricLabel}>{label}</span>}
          {bt ? <span className={styles.pairedValues}>
            {(['bz', 'bt'] as const).map(component => {
              const reading = component === 'bz' ? { point, usable } : bt;
              const delta = snapshot.data?.changes_1h?.[component];
              return <span key={component} className={styles.pairedReading}>
                <span className={styles.pairedLabel}>{component === 'bz' ? 'Bz' : 'Bt'}, nT</span>
                <span className={styles.metricValue}>{number(reading.point?.value)}</span>
                <span className={styles.pairedChange} title={reading.usable ? trend(delta, 'nT') : 'Change unavailable'}>
                  {reading.usable && delta != null ? `${delta > 0 ? '+' : ''}${number(delta, 2)}` : '—'} <span>/ 1h</span>
                </span>
              </span>;
            })}
          </span> : <span className={styles.metricValue}>{number(point?.value, metric === 'kp' ? 2 : 1)} <small>{unit}</small></span>}
          <span className={styles.metricFooter}><span className={styles.metricState} data-warning={snapshot.data && (!usable || bt && !bt.usable)}><span className={styles.dot} />{bt && !bt.usable ? bt.status : status}</span>
          {!bt && <span className={styles.metricTrend}>
            {metric === 'f10_7' ? 'Daily observation' : metric === 'kp' || metric === 'dst' ? <>{metric === 'kp' ? '3-hour' : 'Hourly'} interval · provisional</> :
              <><TrendIcon size={13} aria-hidden="true" />{usable ? trend(change, unit) : 'Change unavailable'}</>}
          </span>}</span>
        </button>;
      })}
    </div>
  </section><ObservationInspector selected={selected} snapshot={snapshot} /></div>;
}
