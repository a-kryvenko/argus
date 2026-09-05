'use client';

import { useEffect, useState } from 'react';
import { apiRequest, type ObservationPoint } from '../_utils/api';
import styles from './page.module.css';

const metrics: { key: keyof Omit<ObservationPoint, 'issue_time'>; label: string; unit: string }[] = [
  { key: 'v', label: 'Solar wind speed', unit: 'km/s' },
  { key: 'n', label: 'Proton density', unit: 'cm⁻³' },
  { key: 't', label: 'Temperature', unit: 'K' },
  { key: 'bz', label: 'Bz', unit: 'nT' },
  { key: 'bx', label: 'Bx', unit: 'nT' },
  { key: 'by', label: 'By', unit: 'nT' },
  { key: 'kp', label: 'Kp index', unit: '' },
  { key: 'dst', label: 'Dst index', unit: 'nT' },
  { key: 'ap', label: 'Ap index', unit: '' },
  { key: 'f10_7', label: 'F10.7 solar flux', unit: 'sfu' },
  { key: 's10', label: 'S10 estimate', unit: '' },
  { key: 'm10', label: 'M10 estimate', unit: '' },
  { key: 'y10', label: 'Y10 estimate', unit: '' },
];

function timestamp(value: string) {
  return new Date(value).toISOString().replace('T', ' ').slice(0, 16);
}

function number(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('en-US', { maximumFractionDigits: 1 }) : '—';
}

export default function Live() {
  const [points, setPoints] = useState<ObservationPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [checkedAt, setCheckedAt] = useState<string>();

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const data = await apiRequest<{ points: ObservationPoint[] }>('/public/observations/history?limit=24', {
          signal: controller.signal,
          cache: 'no-store',
        });
        if (controller.signal.aborted) return;
        setPoints(data.points);
        setCheckedAt(new Date().toISOString());
        setError(false);
      } catch {
        if (controller.signal.aborted) return;
        setError(true);
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
          timer = setTimeout(refresh, 60_000);
        }
      }
    }
    void refresh();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  const latest = points.at(-1);
  const delayed = latest && checkedAt && new Date(checkedAt).getTime() - new Date(latest.issue_time).getTime() > 3 * 3_600_000;

  return (
    <main className={`container ${styles.page}`}>
      <h1>Live observations</h1>
      <p className={styles.description}>Latest available hourly observations. Checks for updates every minute. All times are UTC.</p>
      <p className={styles.description}>These are normalized data; missing measurements may be interpolated or filled during processing.</p>
      <p className={styles.description}>S10, M10 and Y10 are calibrated GOES estimates of daily solar indices, updated from the UTC day’s available observations at each hour boundary. Midnight closes the previous day. These estimates are provisional; no forecasts are shown. A dash means no calibrated estimate is available for that hour; these indices are not gap-filled.</p>
      <nav className="api-links" aria-label="Observations API">
        <a href="/api/v1/public/observations/latest">Latest observation API (JSON)</a>
        <a href="/api/v1/public/observations/history?limit=24">Observation history API (JSON)</a>
      </nav>
      {loading && <p className="state-message" role="status">Loading observations…</p>}
      {error && <p className="state-message" role="alert">Unable to refresh observations. {latest ? 'Showing the last loaded data. ' : ''}Retrying automatically in one minute.</p>}
      {!loading && !error && !latest && <p className="state-message">No observations are available yet.</p>}
      {latest && <>
        <div className={styles.status}>
          <p>Observed: <time dateTime={latest.issue_time}>{timestamp(latest.issue_time)} UTC</time></p>
          {checkedAt && <p>Last checked: {timestamp(checkedAt)} UTC</p>}
          {delayed && <p className={styles.warning}>Latest observation is more than 3 hours old.</p>}
        </div>
        <div className={styles.grid}>
          {metrics.map(metric => <section className={styles.card} key={metric.key}>
            <h2>{metric.label}</h2>
            <p className={styles.value}>{number(latest[metric.key])} <span>{metric.unit}</span></p>
          </section>)}
        </div>
        <h2 className={styles.historyHeading}>Observation history</h2>
        <div className={styles.tableWrapper} tabIndex={0} role="region" aria-label="Observation history, scroll horizontally to see all measurements">
          <table className={styles.table}>
            <caption>Last {points.length} hourly records, newest first</caption>
            <thead><tr><th scope="col">Observed (UTC)</th>{metrics.map(metric => <th scope="col" key={metric.key}>{metric.label}{metric.unit && <small>{metric.unit}</small>}</th>)}</tr></thead>
            <tbody>{[...points].reverse().map(point => <tr key={point.issue_time}>
              <th scope="row"><time dateTime={point.issue_time}>{timestamp(point.issue_time)}</time></th>
              {metrics.map(metric => <td key={metric.key}>{number(point[metric.key])}</td>)}
            </tr>)}</tbody>
          </table>
        </div>
      </>}
    </main>
  );
}
