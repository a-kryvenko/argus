'use client';

import { useMemo } from 'react';
import { ArrowUpRight, Radio } from 'lucide-react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { chartPoints, windUnits, type History, type Metric } from './solarWind';
import { isWind, metricColors, number, windLabels, type Selection } from './observations';
import type { QueryResult } from './useLiveQuery';
import styles from './page.module.css';
import HistoryCoverage from './HistoryCoverage';

const charts: { title: string; unit: string; metrics: Metric[] }[] = [
  { title: 'Magnetic field · GSM Bz and total Bt', unit: 'nT', metrics: ['bz', 'bt'] },
  { title: 'Solar wind speed', unit: 'km/s', metrics: ['v'] },
  { title: 'Proton density', unit: 'cm⁻³', metrics: ['n'] },
];
function timestamp(value: string | number) {
  return new Date(value).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
}

export default function SolarWindLive({ hours, history, selected, onSelect }: {
  hours: number; history: QueryResult<History>; selected: Selection; onSelect: (metric: Selection) => void;
}) {
  const { data: visibleHistory, failed } = history;
  const aggregated = (visibleHistory?.resolution_seconds ?? 60) > 60;
  const visibleCharts = useMemo(() => {
    const base = aggregated ? [
      { title: 'Magnetic field · GSM Bz', unit: 'nT', metrics: ['bz'] as Metric[] },
      { title: 'Total magnetic field Bt', unit: 'nT', metrics: ['bt'] as Metric[] },
      ...charts.slice(1),
    ] : charts;
    return isWind(selected) && !['v', 'n', 'bz', 'bt'].includes(selected)
      ? [...base, { title: windLabels[selected], unit: windUnits[selected], metrics: [selected] }] : base;
  }, [aggregated, selected]);
  const data = useMemo(() => visibleHistory ? visibleCharts.map(chart => chartPoints(visibleHistory, chart.metrics)) : [], [visibleHistory, visibleCharts]);
  const resolution = visibleHistory?.resolution_seconds;
  return <section aria-label="Solar wind at L1" className={styles.chartSection}>
    <div className={styles.sectionHeading}><h2><Radio size={15} aria-hidden="true" />Solar wind <span>L1</span></h2><span>{resolution === undefined ? 'Awaiting history' : resolution === 60 ? '1-minute samples' : `${resolution / 60}-minute means`}</span></div>
    {failed && <p role="alert" className={styles.notice}>Solar wind history could not be refreshed. {visibleHistory ? 'Showing the last response for this period.' : 'No history available for this period.'} Retrying automatically.</p>}
    {!visibleHistory && !failed && <div className={styles.chartPlaceholder} role="status">Loading solar wind history…</div>}
    {visibleHistory && !data.some(points => points.length > 0) && <p className={styles.notice}>No stored measurements in this interval.</p>}
    <div className={styles.chartGrid}>
      {visibleHistory && visibleCharts.map((chart, index) => <section className={`${styles.chart} ${chart.metrics.length > 1 ? styles.wideChart : ''}`} key={chart.title}
        aria-label={`${chart.title}, ${chart.unit}`} data-active={chart.metrics.includes(selected as Metric)}>
        <div className={styles.chartHeading}>
          <h3>{chart.title} <small>{chart.unit}</small></h3>
          <div className={styles.chartActions}>{chart.metrics.map(metric => <button key={metric} onClick={() => onSelect(metric)} aria-label={`Inspect ${windLabels[metric]}`} aria-pressed={selected === metric}>
            {metric.toUpperCase()}<ArrowUpRight size={12} aria-hidden="true" />
          </button>)}</div>
        </div>
        <ResponsiveContainer width="100%" height={chart.metrics.length > 1 ? 240 : 205}>
          <LineChart data={data[index]} syncId="solar-wind" syncMethod="value" margin={{ top: 8, right: 18, left: -8, bottom: 4 }}>
            <CartesianGrid stroke="var(--border-subtle)" vertical={false} />
            <XAxis dataKey="time" type="number" domain={[Date.parse(visibleHistory.from), Date.parse(visibleHistory.to)]}
              tickFormatter={value => new Date(value).toISOString().slice(hours > 24 ? 5 : 11, 16).replace('T', ' ')} minTickGap={45}
              tick={{ fontSize: 11, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} tickLine={false} axisLine={false} />
            <YAxis width={60} domain={['auto', 'auto']} tick={{ fontSize: 11, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} tickFormatter={value => number(value)} tickLine={false} axisLine={false} />
            <Tooltip isAnimationActive={false} labelFormatter={value => timestamp(Number(value))}
              contentStyle={{ background: '#1b232c', borderColor: '#3a4857', borderRadius: 3, color: '#e4eaf0', fontSize: 12 }}
              content={aggregated ? ({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const point = payload[0].payload as Record<string, number | undefined>;
                return <div className={styles.chartTooltip}>
                  <p>{timestamp(Number(label))} · bucket start</p>
                  {chart.metrics.map(metric => <div key={metric}>
                    <strong style={{ color: metricColors[metric] }}>{windLabels[metric]}</strong>
                    {(['min', 'mean', 'max'] as const).map(stat => <p key={stat}>{stat}: {number(point[stat === 'mean' ? metric : `${metric}_${stat}`])} {chart.unit}</p>)}
                    <p>Coverage: {number(point[`${metric}_coverage`])}%</p>
                  </div>)}
                </div>;
              } : undefined} />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} iconType="plainline" />
            {chart.metrics.includes('bz') && <ReferenceLine y={0} stroke="#607080" strokeDasharray="3 3" />}
            {chart.metrics.map(metric => <Line key={metric} dataKey={metric} name={aggregated ? 'Mean' : windLabels[metric]} unit={` ${chart.unit}`}
              stroke={metricColors[metric]} type="linear" legendType="line" dot={false} connectNulls={false} isAnimationActive={false} strokeWidth={1.5} />)}
          </LineChart>
        </ResponsiveContainer>
        <div className={styles.chartCoverage}>{chart.metrics.map(metric => <HistoryCoverage key={metric} label={windLabels[metric]} coverage={visibleHistory.series[metric]?.coverage} processing={visibleHistory.series[metric]?.processing} />)}</div>
      </section>)}
    </div>
    <p className={styles.sectionNote}>{aggregated ? 'Closed UTC windows · min/max and coverage in tooltips · coverage includes all expected minutes.' : 'Native L1 measurements · no Earth-arrival time shift · gaps and spacecraft changes break lines.'}</p>
  </section>;
}
