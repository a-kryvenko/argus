'use client';

import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Globe2 } from 'lucide-react';
import { intervalBounds, indexDefinitions, type IndexHistory, type IndexMetric, type IndexSample } from './geomagnetic';
import { metricColors, number, stamp, type Selection } from './observations';
import type { QueryResult } from './useLiveQuery';
import styles from './page.module.css';
import HistoryCoverage from './HistoryCoverage';

const metrics: IndexMetric[] = ['kp', 'dst'];
function IntervalChart({ metric, history, selected, onSelect }: {
  metric: IndexMetric; history: IndexHistory; selected: Selection; onSelect: (metric: Selection) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(400);
  const [hovered, setHovered] = useState<IndexSample>();
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(entries => setWidth(Math.max(240, entries[0].contentRect.width)));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const series = history.series[metric];
  const { label, unit } = indexDefinitions[metric];
  const start = Date.parse(history.from), end = Date.parse(history.to);
  const values = series.points.filter(p => p.value != null && p.quality !== 'flagged').map(p => p.value as number);
  const low = metric === 'kp' ? 0 : Math.min(0, ...values) - 10;
  const high = metric === 'kp' ? 9 : Math.max(0, ...values) + 10;
  const timeTickCount = width < 500 ? 3 : 4;
  const x = (time: number) => 46 + (time-start)/(end-start)*(width-64);
  const y = (value: number) => 170 - (value-low)/(high-low)*150;
  return <section className={styles.chart} aria-label={`${label} history`} data-active={selected === metric}>
    <div className={styles.chartHeading}><h3>{label}{unit && ` (${unit})`}</h3><div className={styles.chartActions}>
      <button onClick={() => onSelect(metric)} aria-label={`Inspect ${label}`} aria-pressed={selected === metric}>{metric.toUpperCase()}<ArrowUpRight size={12} aria-hidden="true" /></button>
    </div></div>
    {values.length === 0 && <p className={styles.description}>No usable observations in this interval.</p>}
    <div ref={container}>
      <svg width="100%" height={205} viewBox={`0 0 ${width} 205`} role="group" aria-label={`${label}: ${metric === 'kp' ? 'three-hour blocks' : 'hourly segments'}, UTC`}>
        {[0, 1, 2, 3].map(tick => {
          const value = low + (high-low)*tick/3;
          return <g key={tick}><line x1={46} x2={width-18} y1={y(value)} y2={y(value)} stroke="var(--border-subtle)" /><text x={36} y={y(value)+4} textAnchor="end" fill="var(--text-muted)" fontSize={11} fontFamily="var(--font-mono)">{number(value)}</text></g>;
        })}
        <line x1={46} x2={width-18} y1={y(0)} y2={y(0)} stroke="#607080" strokeDasharray="3 3" />
        {Array.from({ length: timeTickCount }, (_, tick) => {
          const time = start + (end-start)*tick/(timeTickCount-1);
          return <text key={tick} x={x(time)} y={195} textAnchor={tick === 0 ? 'start' : tick === timeTickCount-1 ? 'end' : 'middle'} fill="var(--text-muted)" fontSize={11} fontFamily="var(--font-mono)">{new Date(time).toISOString().slice(end-start > 86400000 ? 5 : 11, end-start > 86400000 && width < 500 ? 10 : 16).replace('T', ' ')}</text>;
        })}
        {series.points.map(point => {
          const bounds = intervalBounds(point, start, end);
          if (!bounds || point.value == null) return null;
          const left = x(bounds[0]), right = x(bounds[1]);
          const description = `${number(point.value)} ${unit}, ${stamp(point.interval_start)} to ${stamp(point.interval_end)} UTC, ${point.interval_status}`;
          return <g key={point.interval_start} tabIndex={0} aria-label={description} onFocus={() => setHovered(point)} onBlur={() => setHovered(undefined)} onMouseEnter={() => setHovered(point)} onMouseLeave={() => setHovered(undefined)}>
            <title>{description}</title>
            {metric === 'kp' ? <rect x={left} y={y(point.value)} width={Math.max(0.5, right-left-1)} height={Math.max(1, y(0)-y(point.value))} fill={metricColors[metric]} opacity={0.75} /> : <line x1={left} x2={right} y1={y(point.value)} y2={y(point.value)} stroke={metricColors[metric]} strokeWidth={2} />}
            <rect x={left} y={20} width={Math.max(0.5, right-left)} height={150} fill="transparent" />
          </g>;
        })}
      </svg>
    </div>
    <p className={styles.chartDetail}>{hovered ? `${number(hovered.value)} ${unit} · ${stamp(hovered.interval_start)} – ${stamp(hovered.interval_end)} UTC${hovered.interval_status === 'in_progress' ? ' · in progress' : ''}` : `${metric === 'kp' ? 'Three-hour intervals · preliminary estimate' : 'Hourly intervals · provisional'} · hover or focus to inspect`}</p>
    <div className={styles.chartCoverage}><HistoryCoverage label={label} coverage={series.coverage} /></div>
  </section>;
}

export default function GeomagneticLive({ history, selected, onSelect }: {
  history: QueryResult<IndexHistory>; selected: Selection; onSelect: (metric: Selection) => void;
}) {
  const { data: visible, failed } = history;
  return <section aria-label="Geomagnetic observations" className={styles.chartSection}>
    <div className={styles.sectionHeading}><h2><Globe2 size={15} aria-hidden="true" />Geomagnetic activity <span>EARTH</span></h2><span>Native intervals</span></div>
    {failed && <p role="alert" className={styles.notice}>Index history could not be refreshed. {visible ? 'Showing the last response for this period.' : 'No history available for this period.'} Retrying automatically.</p>}
    <div className={styles.chartGrid}>{visible ? metrics.map(metric => <IntervalChart key={`${metric}-${visible.from}-${visible.to}`} metric={metric} history={visible} selected={selected} onSelect={onSelect} />) : !failed && <div className={styles.chartPlaceholder} role="status">Loading index history…</div>}</div>
    <p className={styles.sectionNote}>Original interval boundaries · blank intervals are missing or flagged · operational values may be revised.</p>
  </section>;
}
