'use client';

import { useEffect, useRef, useState } from 'react';
import { intervalBounds, type IndexHistory, type IndexMetric, type IndexSample } from './geomagnetic';
import { useLiveQuery, type LiveSnapshot } from './useLiveQuery';
import styles from './page.module.css';
import HistoryCoverage from './HistoryCoverage';

const endpoint = '/public/observations/geomagnetic';
const metrics: IndexMetric[] = ['kp', 'dst'];
const stamp = (value: string | number) => new Date(value).toISOString().replace('T', ' ').slice(0, 16);
const number = (value: number | null | undefined) => value == null ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: 2 });

function IntervalChart({ metric, history }: { metric: IndexMetric; history: IndexHistory }) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [hovered, setHovered] = useState<IndexSample>();
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(entries => setWidth(Math.max(260, entries[0].contentRect.width)));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const series = history.series[metric];
  const label = metric === 'kp' ? 'Kp' : series.label;
  const start = Date.parse(history.from), end = Date.parse(history.to);
  const values = series.points.filter(p => p.value != null && p.quality !== 'flagged').map(p => p.value as number);
  const low = metric === 'kp' ? 0 : Math.min(0, ...values) - 10;
  const high = metric === 'kp' ? 9 : Math.max(0, ...values) + 10;
  const timeTickCount = width < 500 ? 3 : 4;
  const x = (time: number) => 48 + (time-start)/(end-start)*(width-64);
  const y = (value: number) => 200 - (value-low)/(high-low)*180;
  return <section className={styles.chart} aria-label={`${label} history`}>
    <h3>{label}{series.unit && ` (${series.unit})`}</h3>
    {metric === 'kp' && <p className={styles.chartDetail}>NOAA SWPC · Preliminary estimate</p>}
    <HistoryCoverage label={label} coverage={series.coverage} />
    {values.length === 0 && <p className={styles.description}>No usable observations in this interval.</p>}
    <div ref={container}>
      <svg width="100%" height={240} viewBox={`0 0 ${width} 240`} role="group" aria-label={`${label}: ${metric === 'kp' ? 'three-hour blocks' : 'hourly segments'}, UTC`}>
        {[0, 1, 2, 3].map(tick => {
          const value = low + (high-low)*tick/3;
          return <g key={tick}><line x1={48} x2={width-16} y1={y(value)} y2={y(value)} stroke="var(--border)" strokeDasharray="3 3" /><text x={40} y={y(value)+4} textAnchor="end" fill="var(--text-secondary)" fontSize={12}>{number(value)}</text></g>;
        })}
        <line x1={48} x2={width-16} y1={y(0)} y2={y(0)} stroke="#777" />
        {Array.from({ length: timeTickCount }, (_, tick) => {
          const time = start + (end-start)*tick/(timeTickCount-1);
          return <text key={tick} x={x(time)} y={224} textAnchor={tick === 0 ? 'start' : tick === timeTickCount-1 ? 'end' : 'middle'} fill="var(--text-secondary)" fontSize={12}>{new Date(time).toISOString().slice(end-start > 86400000 ? 5 : 11, end-start > 86400000 && width < 500 ? 10 : 16).replace('T', ' ')}</text>;
        })}
        {series.points.map(point => {
          const bounds = intervalBounds(point, start, end);
          if (!bounds || point.value == null) return null;
          const left = x(bounds[0]), right = x(bounds[1]);
          const description = `${number(point.value)} ${series.unit}, ${stamp(point.interval_start)} to ${stamp(point.interval_end)} UTC, ${point.interval_status}`;
          return <g key={point.interval_start} tabIndex={0} aria-label={description} onFocus={() => setHovered(point)} onMouseEnter={() => setHovered(point)}>
            <title>{description}</title>
            {metric === 'kp' ? <rect x={left} y={y(point.value)} width={Math.max(0.5, right-left-1)} height={Math.max(1, y(0)-y(point.value))} fill="#a4a9fa" opacity={0.85} /> : <line x1={left} x2={right} y1={y(point.value)} y2={y(point.value)} stroke="#68d5c5" strokeWidth={3} />}
            <rect x={left} y={20} width={Math.max(0.5, right-left)} height={180} fill="transparent" />
          </g>;
        })}
      </svg>
    </div>
    <p className={styles.chartDetail}>{hovered ? `${number(hovered.value)} ${series.unit} · ${stamp(hovered.interval_start)} – ${stamp(hovered.interval_end)} UTC${hovered.interval_status === 'in_progress' ? ' · interval in progress' : ''}` : 'Hover or focus an interval for its value and UTC times.'}</p>
  </section>;
}

export default function GeomagneticLive({ hours, onHoursChange, snapshot }: { hours: number; onHoursChange: (hours: number) => void; snapshot: LiveSnapshot }) {
  const { data: visible, failed } = useLiveQuery<IndexHistory>(`${endpoint}/history`, hours);
  const { receivedAt: checkedAt, now } = snapshot;
  const latest = snapshot.data ? { generated_at: snapshot.data.generated_at, series: snapshot.data.geomagnetic } : undefined;
  const errors = [snapshot.failed ? 'Latest indices could not be refreshed.' : '', failed ? 'Index history could not be refreshed.' : ''].filter(Boolean);
  return <section aria-label="Geomagnetic observations">
    <h2>Geomagnetic activity</h2>
    <p className={styles.description}>Kp retains its three-hour intervals. Kyoto Dst is shown hourly. Both are preliminary operational data and may be revised.</p>
    {errors.length > 0 && <p role="alert" className={styles.warning}>{errors.join(' ')} Showing available saved responses; retrying in one minute.</p>}
    <details className={styles.additional}>
      <summary>Geomagnetic measurement details</summary>
      <div className={styles.grid}>{metrics.map(metric => {
      const series = latest?.series[metric], point = series?.latest;
      const elapsed = now && checkedAt ? Math.max(0, (now-checkedAt)/1000) : 0;
      const serverTime = latest ? Date.parse(latest.generated_at) + elapsed*1000 : 0;
      const lag = point ? Math.max(0, (serverTime-Date.parse(point.interval_end))/1000) : 0;
      const stale = series && lag > series.stale_after_seconds;
      return <section key={metric} className={styles.card}>
        <h3>{metric === 'kp' ? 'Kp' : 'Real-time Dst'}</h3>
        {metric === 'kp' && <p className={styles.sampleTime}>NOAA SWPC · Preliminary estimate</p>}
        <p className={styles.value}>{number(point?.value)} <span>{series?.unit}</span></p>
        <p className={point?.value == null || stale || point.quality === 'flagged' ? styles.warning : styles.fresh}>{!latest ? 'Loading…' : point?.value == null ? 'Unavailable' : stale ? 'Delayed' : 'Latest available interval'}</p>
        {point && <>
          <p className={styles.sampleTime}>{stamp(point.interval_start)} – {stamp(point.interval_end)} UTC</p>
          <p className={styles.sampleTime}>{point.quality === 'flagged' ? 'Provider quality issue' : serverTime < Date.parse(point.interval_end) ? 'Interval in progress' : 'Completed interval · subject to revision'}</p>
          {point.station_count != null && <p className={styles.sampleTime}>{point.station_count} contributing stations</p>}
        </>}
        <p className={styles.sampleTime}>{series?.source}</p>
      </section>;
    })}</div>
    </details>
    <nav className="api-links" aria-label="Geomagnetic API">
      <a href={`/api/v1${endpoint}/latest`}>Latest indices · JSON</a>
      <a href={`/api/v1${endpoint}/history`}>Index history · JSON</a>
      <a href="https://www.spaceweather.gov/products/planetary-k-index">NOAA Kp</a>
      <a href="https://wdc.kugi.kyoto-u.ac.jp/dstdir/">Kyoto Dst</a>
    </nav>
    <div className={styles.chartHeader}><h3>Index history</h3><div className={styles.periods} role="group" aria-label="Shared history period">{[6,24,72,168,720].map(period => <button key={period} aria-pressed={hours===period} onClick={() => onHoursChange(period)}>{period < 48 ? `${period} hours` : `${period/24} days`}</button>)}</div></div>
    {visible ? metrics.map(metric => <IntervalChart key={`${metric}-${hours}`} metric={metric} history={visible} />) : <p role="status">{errors.length ? 'History unavailable for this period.' : 'Loading index history…'}</p>}
    <p className={styles.description}>Blank intervals indicate missing or flagged data. Freshness allows for publication delay after an interval ends: four hours for Kp, two hours for Dst (the next native interval plus one hour).</p>
  </section>;
}
