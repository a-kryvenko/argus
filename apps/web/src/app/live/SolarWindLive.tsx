'use client';

import { useMemo } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { chartPoints, type History, type Metric } from './solarWind';
import { metricColors, number, windLabels } from './observations';
import type { QueryResult } from './useLiveQuery';
import styles from './page.module.css';
import HistoryCoverage from './HistoryCoverage';

const charts: { title: string; unit: string; metrics: Metric[] }[] = [
  { title: 'Magnetic field · Bz / Bt', unit: 'nT', metrics: ['bz', 'bt'] },
  { title: 'Solar wind speed', unit: 'km/s', metrics: ['v'] },
  { title: 'Proton density', unit: 'cm⁻³', metrics: ['n'] },
];
function timestamp(value: string | number) {
  return new Date(value).toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
}

export default function SolarWindLive({ hours, history }: {
  hours: number; history: QueryResult<History>;
}) {
  const { data: visibleHistory, failed } = history;
  const aggregated = (visibleHistory?.resolution_seconds ?? 60) > 60;
  const data = useMemo(() => visibleHistory ? charts.map(chart => chartPoints(visibleHistory, chart.metrics)) : [], [visibleHistory]);
  const resolution = visibleHistory?.resolution_seconds;
  return <section aria-label="Solar wind at L1" className={styles.chartSection}>
    <div className={styles.sectionHeading}><h3>Solar activity <span>L1</span></h3><span>{resolution === undefined ? 'Awaiting history' : resolution === 60 ? '1-minute samples' : resolution === 3600 ? 'Hourly means' : `${resolution / 60}-minute means`}</span></div>
    {failed && <p role="alert" className={styles.notice}>Solar wind history could not be refreshed. {visibleHistory ? 'Showing the last response for this period.' : 'No history available for this period.'} Retrying automatically.</p>}
    {!visibleHistory && !failed && <div className={styles.chartPlaceholder} role="status">Loading solar wind history…</div>}
    {visibleHistory && !data.some(points => points.length > 0) && <p className={styles.notice}>No stored measurements in this interval.</p>}
    <div className={`${styles.chartGrid} ${styles.windChartGrid}`}>
      {visibleHistory && charts.map((chart, index) => <section className={styles.chart} key={chart.title}
        aria-label={`${chart.title}, ${chart.unit}`}>
        <div className={styles.chartHeading}>
          <h4>{chart.title} <small>{chart.unit}</small></h4>

        </div>
        <ResponsiveContainer width="100%" height={190}>
          <LineChart data={data[index]} syncId="solar-wind" syncMethod="value" margin={{ top: 8, right: 18, left: -8, bottom: 4 }}>
            <CartesianGrid stroke="var(--border-subtle)" vertical={false} />
            <XAxis dataKey="time" type="number" domain={[Date.parse(visibleHistory.from), Date.parse(visibleHistory.to)]}
              tickFormatter={value => new Date(value).toISOString().slice(hours > 24 ? 5 : 11, 16).replace('T', ' ')} minTickGap={45}
              tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} tickLine={false} axisLine={false} />
            <YAxis width={48} domain={['auto', 'auto']} tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} tickFormatter={value => number(value)} tickLine={false} axisLine={false} />
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
            {chart.metrics.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} height={22} iconType="plainline" />}
            {chart.metrics.includes('bz') && <ReferenceLine y={0} stroke="#607080" strokeDasharray="3 3" ifOverflow="extendDomain" />}
            {chart.metrics.map(metric => <Line key={metric} dataKey={metric} name={metric === 'bz' ? 'Bz · GSM' : metric === 'bt' ? 'Bt · total' : windLabels[metric]} unit={` ${chart.unit}`}
              stroke={metricColors[metric]} type="linear" legendType="line" dot={false} connectNulls={false} isAnimationActive={false} strokeDasharray={metric === 'bt' ? '5 3' : undefined} strokeWidth={metric === 'bz' ? 1.8 : 1.5} />)}
          </LineChart>
        </ResponsiveContainer>
        <details className={styles.chartDataDetails}>
          <summary><span>Data coverage</span><span>{chart.metrics.map(metric => {
            const percent = visibleHistory.series[metric]?.coverage.percent;
            return `${chart.metrics.length > 1 ? `${metric.toUpperCase()} ` : ''}${percent == null ? '—' : `${number(percent)}%`}`;
          }).join(' · ')}</span></summary>
          <div className={styles.chartCoverage}>{chart.metrics.map(metric => <HistoryCoverage key={metric} label={windLabels[metric]} coverage={visibleHistory.series[metric]?.coverage} processing={visibleHistory.series[metric]?.processing} />)}</div>
        </details>
      </section>)}
    </div>
    <p className={styles.sectionNote}>{aggregated ? 'Closed UTC windows · min/max and coverage in tooltips · coverage includes all expected minutes.' : 'Native L1 measurements · no Earth-arrival time shift · gaps and spacecraft changes break lines.'}</p>
  </section>;
}
