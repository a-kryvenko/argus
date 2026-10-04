'use client';
import MetricValue from './MetricValue';
import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Line } from 'recharts';
import { type VariableMetrics } from '../_utils/transform';
import { colors } from './MetricChart';
import styles from '../../_components/forecast.module.css';
import local from './metrics.module.css';

export default function ReliabilityChart({ data, hour, labels }: { data: VariableMetrics; hour: number | undefined; labels: Record<string, string> }) {
  const series = data.binary.map(item => ({ key: String(item.threshold), points: item.by_lead_hour.find(row => row.lead_hours === hour)?.reliability ?? [] }));
  const bins = [...new Set(series.flatMap(item => item.points.map(point => point.predicted_probability)))].sort((a, b) => a - b);
  const rows = bins.map(predicted => ({ predicted, ...Object.fromEntries(series.map(item => [item.key, item.points.find(point => point.predicted_probability === predicted)?.observed_frequency ?? null])) }));
  return <section className={styles.chart} aria-label="Reliability at selected lead hour">
    <div className={styles.chartHeading}><h3>Reliability<small>{hour == null ? 'No lead selected' : `Lead +${hour}h`}</small></h3><span className={styles.chartTag}>CALIBRATION</span></div>
    <div className={local.legend}><span><i style={{ background: '#8797a8' }} />Perfect calibration</span>{Object.entries(labels).map(([key, label], i) => <span key={key}><i style={{ background: colors[i % colors.length] }} />{label}</span>)}</div>
    {bins.length ? <div className={local.reliabilityPlot}><ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows} margin={{ top: 12, right: 28, bottom: 28, left: 8 }}>
        <CartesianGrid stroke="#26333f" />
        <XAxis dataKey="predicted" type="number" domain={[0, 1]} tickFormatter={value => `${Math.round(value * 100)}%`} tick={{ fill: '#92a4b5', fontSize: 10 }} tickLine={false} axisLine={false} label={{ value: 'Predicted probability', position: 'insideBottom', offset: -16, fill: '#92a4b5', fontSize: 10 }} />
        <YAxis type="number" domain={[0, 1]} width={54} tickFormatter={value => `${Math.round(value * 100)}%`} tick={{ fill: '#92a4b5', fontSize: 10 }} tickLine={false} axisLine={false} />
        <Tooltip isAnimationActive={false} content={({ active, payload }) => {
          const row = payload?.find(item => item.dataKey !== 'perfect')?.payload as Record<string, number | null> | undefined;
          return active && row ? <div className={styles.tooltip}><strong>Predicted probability <MetricValue value={(row.predicted ?? 0) * 100} />%</strong>{Object.entries(labels).map(([key, label]) => <p key={key}>{label}<b>{row[key] == null ? '—' : <><MetricValue value={row[key] * 100} />%</>}</b></p>)}</div> : null;
        }} />
        <Line data={[{ predicted: 0, perfect: 0 }, { predicted: 1, perfect: 1 }]} dataKey="perfect" stroke="#8797a8" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
        {Object.entries(labels).map(([key, label], i) => <Line key={key} dataKey={key} name={label} stroke={colors[i % colors.length]} strokeWidth={1.6} dot={{ r: 3, fill: colors[i % colors.length] }} connectNulls={false} isAnimationActive={false} />)}
      </LineChart>
    </ResponsiveContainer></div> : <p className={styles.emptyChart} role="status">No reliability points at this lead hour.</p>}
    <p className={styles.chartNote}>Vertical axis: observed event frequency. The diagonal represents perfect calibration. Values apply to the selected lead hour.</p>
    {bins.length > 0 && <details className={local.reliabilityTable}><summary>Calibration values</summary><div className={styles.tableScroll}><table><caption>Predicted probabilities and observed frequencies as fractions (0–1).</caption><thead><tr><th scope="col">Predicted</th>{Object.entries(labels).map(([key, label]) => <th scope="col" key={key}>{label}</th>)}</tr></thead><tbody>{rows.map(row => <tr key={row.predicted}><th scope="row"><MetricValue value={row.predicted} /></th>{Object.keys(labels).map(key => <td key={key}><MetricValue value={(row as Record<string, number | null>)[key]} /></td>)}</tr>)}</tbody></table></div></details>}
  </section>;
}
