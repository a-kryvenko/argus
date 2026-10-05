'use client';
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatForecastTime, quantileData } from '../_utils/forecast';
import styles from './forecast.module.css';

type QuantileRow = ReturnType<typeof quantileData>[number] & { timestamp: number; band: [number, number] | null; observed?: number | null; actual?: number | null };
const number = (value: number | null | undefined) => value == null ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: 3 });

export default function WindChart({ data, title = 'Solar wind speed', unit = 'km/s', selectedTime, onSelectTime, now, eventTime, comparison = false }: {
  data: Array<ReturnType<typeof quantileData>[number] & { observed?: number | null; actual?: number | null }>; now?: number; eventTime?: number; comparison?: boolean; title?: string; unit?: string; selectedTime?: string; onSelectTime?: (time: string) => void;
}) {
  const rows: QuantileRow[] = data.map(row => ({ ...row, timestamp: Date.parse(row.time),
    band: row.low != null && row.high != null ? [row.low, row.high] : null }));
  const hasQuantiles = !comparison || rows.some(row => row.median != null);
  const available = rows.some(row => row.median != null || row.observed != null || row.actual != null);
  return <section className={styles.chart} aria-label={`${title} ${comparison ? "forecast and observations" : "quantile forecast"}`}>
    <div className={styles.chartHeading}><h3>{title}<small>{unit}</small></h3><span className={styles.chartTag}>{comparison ? "FORECAST / OBSERVED" : "QUANTILES"}</span></div>
    <div className={styles.chartLegend}>{hasQuantiles && <><span><i className={styles.lineSwatch} />Median · q50</span><span><i className={styles.bandSwatch} />q10–q90 interval</span></>}<span>UTC</span>{comparison && <><span style={{ color: "#73c9b0" }}>━ Observed history</span><span style={{ color: "#f1c777" }}>┄ Actual outcome</span></>}</div>
    {available ? <div className={styles.quantilePlot}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 12, right: 42, bottom: 8, left: 0 }}
          onClick={event => { const index = Number(event.activeTooltipIndex); if (event.activeTooltipIndex != null && rows[index]) onSelectTime?.(rows[index].time); }}>
          <CartesianGrid vertical={false} stroke="#26333f" />
          <XAxis dataKey="timestamp" type="number" domain={rows.length === 1 ? [rows[0].timestamp - 3600000, rows[0].timestamp + 3600000] : ['dataMin', 'dataMax']}
            minTickGap={50} tick={{ fill: '#92a4b5', fontSize: 10, fontFamily: 'monospace' }} tickLine={false} axisLine={false}
            tickFormatter={value => new Date(value).toISOString().slice(5, 16).replace('T', ' ')} />
          <YAxis width={62} domain={['auto', 'auto']} tick={{ fill: '#92a4b5', fontSize: 10, fontFamily: 'monospace' }} tickLine={false} axisLine={false} tickFormatter={number} />
          <Tooltip isAnimationActive={false} content={({ active, payload }) => {
            const row = payload?.[0]?.payload as QuantileRow | undefined;
            return active && row ? <div className={styles.tooltip}><strong>{formatForecastTime(row.time)}</strong>
              {hasQuantiles && <><p>q50 <b>{number(row.median)} {unit}</b></p><p>q10 <span>{number(row.low)} {unit}</span></p><p>q90 <span>{number(row.high)} {unit}</span></p></>}
              {comparison && <><p>Observed <b>{number(row.observed ?? row.actual)} {unit}</b></p>{hasQuantiles && <p>q50 − observed <b>{number(row.median != null && (row.observed ?? row.actual) != null ? row.median - (row.observed ?? row.actual)! : null)} {unit}</b></p>}</>}
            </div> : null;
          }} />
          <Area dataKey="band" type="linear" fill="#65b5eb" fillOpacity={.13} stroke="#54778e" strokeWidth={.6} connectNulls={false} isAnimationActive={false} name="q10–q90" />
          <Line dataKey="median" type="linear" stroke="#79b9e6" strokeWidth={1.8} dot={false} activeDot={{ r: 4, stroke: '#cbe6fa', strokeWidth: 1 }} connectNulls={false} isAnimationActive={false} name="Median" />
          {comparison && <Line dataKey="observed" type="linear" stroke="#73c9b0" strokeWidth={1.8} dot={false} connectNulls={false} isAnimationActive={false} name="Observed history" />}
          {comparison && <Line dataKey="actual" type="linear" stroke="#f1c777" strokeDasharray="5 3" strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} name="Actual outcome" />}
          {now != null && <ReferenceLine x={now} stroke="#e6edf3" label={{ value: 'Now', fill: '#e6edf3', fontSize: 11 }} />}
          {eventTime != null && <ReferenceLine x={eventTime} stroke="#f1c777" strokeDasharray="4 4" label={{ value: 'T0', fill: '#f1c777', fontSize: 11 }} />}
          {selectedTime && <ReferenceLine x={Date.parse(selectedTime)} stroke="#8999a9" strokeDasharray="3 3" />}
        </ComposedChart>
      </ResponsiveContainer>
    </div> : <p className={styles.emptyChart} role="status">No quantile forecast available in this horizon.</p>}
    <p className={styles.chartNote}>{hasQuantiles ? "Shaded interval: 10th–90th percentiles, not a guaranteed range. Click a time to inspect; missing values remain gaps." : "Observed values in physical units. Missing observations remain gaps; probabilities are shown separately when available."}</p>
  </section>;
}
