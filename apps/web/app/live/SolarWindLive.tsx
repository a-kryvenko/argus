'use client';

import { useEffect, useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { apiRequest } from '../_utils/api';
import { chartPoints, type History, type Latest, type Metric } from './solarWind';
import styles from './page.module.css';

const metrics: Metric[] = ['v', 'n', 'bz', 'bt', 'bx', 'by', 't'];
const labels: Record<Metric, string> = { v: 'Solar wind speed', n: 'Proton density', bz: 'Bz', bt: 'Total field Bt', bx: 'Bx', by: 'By', t: 'Proton temperature' };
const charts: { title: string; unit: string; metrics: Metric[]; colors: string[] }[] = [
  { title: 'Magnetic field · GSM Bz and total Bt', unit: 'nT', metrics: ['bz', 'bt'], colors: ['#8aa4ff', '#f6bd60'] },
  { title: 'Solar wind speed', unit: 'km/s', metrics: ['v'], colors: ['#68d5c5'] },
  { title: 'Proton density', unit: 'cm⁻³', metrics: ['n'], colors: ['#e6a1df'] },
];
const endpoint = '/public/observations/solar-wind';
function timestamp(value: string | number) {
  return new Date(value).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
}
function number(value: number | null | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('en-US', { maximumFractionDigits: 1 }) : '—';
}

export default function SolarWindLive({ hours, onHoursChange }: { hours: number; onHoursChange: (hours: number) => void }) {
  const [latest, setLatest] = useState<Latest>();
  const [history, setHistory] = useState<{ data: History; hours: number }>();
  const [errors, setErrors] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [checkedAt, setCheckedAt] = useState<number>();
  const [now, setNow] = useState<number>();

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      const end = new Date();
      const start = new Date(end.getTime() - hours * 3_600_000);
      const query = new URLSearchParams({ from: start.toISOString(), to: end.toISOString(), metrics: 'bz,bt,v,n' });
      const options = { signal: controller.signal, cache: 'no-store' as const };
      const [current, past] = await Promise.allSettled([
        apiRequest<Latest>(`${endpoint}/latest`, options),
        apiRequest<History>(`${endpoint}/history?${query}`, options),
      ]);
      if (controller.signal.aborted) return;
      const failures: string[] = [];
      if (current.status === 'fulfilled') { setLatest(current.value); setCheckedAt(Date.now()); }
      else failures.push('Latest measurements could not be refreshed.');
      if (past.status === 'fulfilled') setHistory({ data: past.value, hours });
      else failures.push('History could not be refreshed.');
      setErrors(failures);
      setLoading(false);
      setNow(Date.now());
      timer = setTimeout(refresh, 60_000);
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [hours]);

  const visibleHistory = history?.hours === hours ? history.data : undefined;
  const data = useMemo(() => visibleHistory ? charts.map(chart => chartPoints(visibleHistory, chart.metrics)) : [], [visibleHistory]);
  const hasMeasurements = latest && Object.values(latest.series).some(series => series.latest?.value != null);
  function card(metric: Metric) {
    const series = latest?.series[metric];
    const point = series?.latest;
    const elapsed = now && checkedAt ? Math.max(0, Math.floor((now - checkedAt) / 1000)) : 0;
    const age = series?.age_seconds == null ? undefined : series.age_seconds + elapsed;
    const stale = age !== undefined && age > (series?.stale_after_seconds ?? 600);
    return <section className={styles.card} key={metric}>
      <h3>{labels[metric]}{series?.coordinate_system && <small> · {series.coordinate_system}</small>}</h3>
      <p className={styles.value}>{number(point?.value)} <span>{series?.unit}</span></p>
      <p className={point?.value == null || stale || point.quality === 'flagged' ? styles.warning : styles.fresh}>
        {point?.value == null ? 'Measurement unavailable' : stale ? 'Delayed' : 'Recent'}
        {age !== undefined && ` · ${Math.floor(age / 60)} min old`}
      </p>
      {point && <>
        <p className={styles.sampleTime}><time dateTime={point.observed_at}>{timestamp(point.observed_at)}</time></p>
        <p className={styles.sampleTime}>{point.spacecraft} · {point.quality === 'flagged' ? 'Provider quality flag' : 'Operational data'}</p>
      </>}
    </section>;
  }

  return <section aria-label="Solar wind at L1">
    <p className={styles.description}>Solar wind measured at L1, before it reaches Earth. One-minute source samples; checks for updates every minute. All times are UTC.</p>
    <p className={styles.description}>NOAA selects the active spacecraft separately for plasma and magnetic field. Measurements are not shifted to Earth arrival time. Gaps and flagged samples are not connected on charts.</p>
    <nav className="api-links" aria-label="Solar wind API">
      <a href={`/api/v1${endpoint}/latest`}>Latest solar wind · JSON</a>
      <a href={`/api/v1${endpoint}/history?metrics=bz,bt,v,n`}>24-hour history · JSON</a>
      <a href="https://www.spaceweather.gov/products/solar-wind" target="_blank" rel="noreferrer">NOAA source and methodology</a>
    </nav>
    {loading && <p role="status">Loading solar wind observations…</p>}
    {errors.length > 0 && <p role="alert" className={styles.warning}>{errors.join(' ')} Previously loaded data remain visible. Retrying in one minute.</p>}
    {!loading && !hasMeasurements && <p role="status">No solar wind measurements are available yet.</p>}
    <details className={styles.additional}>
      <summary>Solar wind measurement details</summary>
      {checkedAt && <p className={styles.description}>Latest API check: {timestamp(checkedAt)}. Freshness below uses measurement time.</p>}
      <div className={styles.grid}>{metrics.map(card)}</div>
    </details>
    <div className={styles.chartHeader}>
      <h2>Solar wind history</h2>
      <div className={styles.periods} role="group" aria-label="History period">
        {[6, 24, 72, 168].map(period => <button key={period} aria-pressed={hours === period} onClick={() => onHoursChange(period)}>{period < 48 ? `${period} hours` : `${period / 24} days`}</button>)}
      </div>
    </div>
    {!visibleHistory && <p role="status">{errors.length ? 'History is unavailable for this period.' : 'Loading history…'}</p>}
    {visibleHistory && !data.some(points => points.length > 0) && <p>No stored measurements in this interval.</p>}
    {visibleHistory && data.some(points => points.length > 0) && charts.map((chart, index) => <section className={styles.chart} key={chart.title} aria-label={`${chart.title}, ${chart.unit}`}>
      <h3>{chart.title} <small>({chart.unit})</small></h3>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data[index]} syncId="solar-wind" syncMethod="value" margin={{ top: 8, right: 18, left: 0, bottom: 8 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="time" type="number" domain={[Date.parse(visibleHistory.from), Date.parse(visibleHistory.to)]} tickFormatter={value => new Date(value).toISOString().slice(hours > 24 ? 5 : 11, 16).replace('T', ' ')} minTickGap={45} tick={{ fontSize: 12 }} />
          <YAxis width={64} domain={['auto', 'auto']} tick={{ fontSize: 12 }} tickFormatter={value => number(value)} />
          <Tooltip isAnimationActive={false} labelFormatter={value => timestamp(Number(value))} contentStyle={{ background: '#18181b', borderColor: '#3f3f46', color: '#fff' }} />
          <Legend />
          {chart.metrics.includes('bz') && <ReferenceLine y={0} stroke="#808080" />}
          {chart.metrics.map((metric, index) => <Line key={metric} dataKey={metric} name={labels[metric]} unit={` ${chart.unit}`} stroke={chart.colors[index]} type="linear" dot={false} connectNulls={false} isAnimationActive={false} strokeWidth={1.5} />)}
        </LineChart>
      </ResponsiveContainer>
    </section>)}
    <p className={styles.description}>Values are operational observations, not independently validated by Argus. A delay above 10 minutes is marked “Delayed”. History is shown at its original one-minute resolution, without interpolation.</p>
  </section>;
}
