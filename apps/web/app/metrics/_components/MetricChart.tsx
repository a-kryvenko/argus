'use client';
import MetricValue from './MetricValue';
import { useMemo } from 'react';
import { CartesianGrid, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, ReferenceLine } from 'recharts';
import { chartRows, metricNumber, type MetricRow } from '../_utils/transform';
import styles from '../../_components/forecast.module.css';
import local from './metrics.module.css';

export const colors = ['#79b9e6', '#74b8a7', '#d5b376', '#b49cd1', '#d59191'];
export default function MetricChart({ data, title, labels, unit = '', note, selectedHour, onSelectHour }: {
  data: MetricRow[]; title: string; labels: Record<string, string>; unit?: string; note: string;
  selectedHour: number | undefined; onSelectHour: (hour: number) => void;
}) {
  const rows = useMemo(() => chartRows(data), [data]);
  const hasValues = rows.some(row => Object.keys(labels).some(key => row.values[key] != null));
  return <section className={styles.chart} aria-label={`${title} by lead hour`}>
    <div className={styles.chartHeading}><h3>{title}<small>{unit}</small></h3><span className={styles.chartTag}>BY LEAD HOUR</span></div>
    <div className={local.legend}>{Object.entries(labels).map(([key, label], i) => <span key={key}><i style={{ background: colors[i % colors.length] }} />{label}</span>)}</div>
    {hasValues ? <div className={styles.quantilePlot}><ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows} margin={{ top: 12, right: 28, bottom: 12, left: 0 }} onClick={event => {
        const lead = Number(event.activeLabel);
        if (event.activeLabel != null && data.some(row => row.lead_hours === lead)) onSelectHour(lead);
      }}>
        <CartesianGrid vertical={false} stroke="#26333f" />
        <XAxis dataKey="lead_hours" type="number" domain={rows.length === 1 ? [Math.max(0, rows[0].lead_hours - 1), rows[0].lead_hours + 1] : ['dataMin', 'dataMax']} allowDecimals={false} tickFormatter={value => `+${value}h`} tick={{ fill: '#92a4b5', fontSize: 10 }} tickLine={false} axisLine={false} />
        <YAxis width={62} domain={['auto', 'auto']} tickFormatter={metricNumber} tick={{ fill: '#92a4b5', fontSize: 10 }} tickLine={false} axisLine={false} />
        <Tooltip isAnimationActive={false} content={({ active, payload }) => {
          const row = payload?.[0]?.payload as MetricRow | undefined;
          return active && row ? <div className={styles.tooltip}><strong>Lead +{row.lead_hours} hours</strong>{Object.entries(labels).map(([key, label]) => <p key={key}>{label}<b><MetricValue value={row.values[key]} /></b></p>)}</div> : null;
        }} />
        {Object.entries(labels).map(([key, label], i) => <Line key={key} name={label} dataKey={row => row.values[key] ?? null} type="linear" stroke={colors[i % colors.length]} strokeWidth={1.8}
          dot={({ cx, cy, payload }: { cx?: number; cy?: number; payload?: MetricRow }) => cx == null || cy == null || payload?.values[key] == null ? <g /> : <circle className="recharts-line-dot" cx={cx} cy={cy} r={2} fill={colors[i % colors.length]} onClick={event => { event.stopPropagation(); onSelectHour(payload.lead_hours); }} />}
          activeDot={{ r: 4, pointerEvents: 'none' }} connectNulls={false} isAnimationActive={false} />)}
        {selectedHour != null && <ReferenceLine x={selectedHour} stroke="#8999a9" strokeDasharray="3 3" />}
      </LineChart>
    </ResponsiveContainer></div> : <p className={styles.emptyChart} role="status">No values for this metric.</p>}
    <p className={styles.chartNote}>{note} Click a lead hour to inspect. Missing values remain gaps.</p>
  </section>;
}
