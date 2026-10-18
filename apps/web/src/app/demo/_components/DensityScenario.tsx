'use client';
import { useState } from 'react';
import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { formatForecastTime } from '../../../forecast/format';
import { densityRelease, densityRows, visibleDensityJournal, type DensityScenario as Scenario, type DensityVariant } from '../_lib/density';
import styles from './density.module.css';

const labels = { within_reference: 'Below review threshold', review: 'Review', high: 'High', not_assessed: 'Not assessed' };
const percent = (value: number | null) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;

export default function DensityScenario({ scenario, now, eventTime, showFuture, interactive }: { scenario: Scenario | null; now: number; eventTime: number; showFuture: boolean; interactive: boolean }) {
  const [model, setModel] = useState('jb2008');
  const [drivers, setDrivers] = useState('fixed');
  const [threshold, setThreshold] = useState('2');
  if (!scenario) return <p role="status">The LEO density scenario has not been prepared for this demo.</p>;
  const variant = `${model}_${drivers}` as DensityVariant;
  const release = densityRelease(scenario, variant, now);
  const rows = densityRows(scenario, release, now, showFuture);
  const journal = visibleDensityJournal(scenario, variant, now);
  const scores = scenario.scores.filter(score => score.variant === variant && score.threshold_ratio === Number(threshold));
  const events = scenario.events.filter(event => event.variant === variant && event.threshold_ratio === Number(threshold));
  return <div className={styles.root}>
    <p>GRACE-C · Local density along the measured trajectory · Research scenario</p>
    <div className={styles.controls}>
      <label>Atmosphere model<select disabled={!interactive} value={model} onChange={event => setModel(event.target.value)}><option value="jb2008">JB2008</option><option value="dtm2020">DTM2020</option></select></label>
      <label>Geomagnetic drivers<select disabled={!interactive} value={drivers} onChange={event => setDrivers(event.target.value)}><option value="fixed">Fixed at issue time</option><option value="forecast">Forecast medians</option></select></label>
      <strong role="status" aria-label="Density attention">{release ? labels[release.attention] : 'No release for this hour'}</strong>
    </div>
    <p>{drivers === 'fixed' ? 'Solar and geomagnetic drivers stay fixed at the issue time.' : model === 'jb2008' ? 'Solar drivers stay fixed; forecast Ap/Dst medians drive the experimental geomagnetic response.' : 'Solar drivers stay fixed; forecast Ap medians drive Kp and its trailing mean.'}</p>
    <p>Reference: {scenario.reference.density_kg_m3.toExponential(3)} kg/m³, median of {scenario.reference.samples} observations during the 72 hours before replay. Review ≥2×; high ≥4×.</p>
    <section aria-label="Atmospheric density forecast and observations" className={styles.chart}>
      <h2>Density relative to the shared reference</h2>
      <p>Blue: selected forecast · Green: observed history · Gold: actual future{!showFuture && ' (hidden)'}</p>
      <div className={styles.plot}><ResponsiveContainer width="100%" height="100%"><ComposedChart data={rows} margin={{ top: 20, right: 35, left: 0, bottom: 5 }}>
        <CartesianGrid vertical={false} stroke="#26333f" />
        <XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} minTickGap={50} tickFormatter={value => new Date(value).toISOString().slice(5, 16).replace('T', ' ')} tick={{ fill: '#92a4b5', fontSize: 10 }} />
        <YAxis domain={[0, 'auto']} tick={{ fill: '#92a4b5', fontSize: 11 }} tickFormatter={value => `${value}×`} />
        <Tooltip labelFormatter={value => formatForecastTime(new Date(Number(value)).toISOString())} formatter={value => value == null ? '—' : `${Number(value).toFixed(2)}×`} />
        <Line dataKey="forecast" name="Forecast" stroke="#79b9e6" dot={false} connectNulls={false} isAnimationActive={false} />
        <Line dataKey="observed" name="Observed history" stroke="#73c9b0" dot={false} connectNulls={false} isAnimationActive={false} />
        <Line dataKey="actual" name="Actual future" stroke="#f1c777" strokeDasharray="5 3" dot={false} connectNulls={false} isAnimationActive={false} />
        <ReferenceLine y={2} ifOverflow="extendDomain" stroke="#f1c777" strokeDasharray="3 3" label={{ value: 'Review 2×', fill: '#f1c777', fontSize: 10 }} />
        <ReferenceLine y={4} ifOverflow="extendDomain" stroke="#f17c77" strokeDasharray="3 3" label={{ value: 'High 4×', fill: '#f17c77', fontSize: 10 }} />
        <ReferenceLine x={now} stroke="#e6edf3" label={{ value: 'Now', fill: '#e6edf3', fontSize: 11 }} />
        <ReferenceLine x={eventTime} stroke="#f1c777" strokeDasharray="4 4" />
      </ComposedChart></ResponsiveContainer></div>
    </section>
    <section><h2>Warning sample spans · Selected issue</h2>
      {release?.intervals.length ? <div className={styles.scroll}><table><thead><tr><th>Start UTC</th><th>End UTC</th><th>Attention</th><th>Hourly samples</th><th>Peak ratio</th></tr></thead><tbody>{release.intervals.map(interval => <tr key={interval.start}><td>{formatForecastTime(interval.start)}</td><td>{formatForecastTime(interval.end)}</td><td>{labels[interval.attention]}</td><td>{interval.sample_count}</td><td>{interval.peak_ratio.toFixed(2)}×</td></tr>)}</tbody></table></div> : <p>{release ? 'No warning samples in the selected forecast.' : 'Forecast unavailable.'}</p>}
      <p>Missing samples break spans. Hourly samples do not establish precise crossings or continuous exposure.</p>
    </section>
    <section><h2>Forecast revisions · Through simulated now</h2><div className={styles.scroll}><table><thead><tr><th>Issue UTC</th><th>Attention</th><th>Warning spans</th><th>Removed warning samples</th><th>Missing samples</th></tr></thead><tbody>{journal.map(entry => <tr key={entry.issue_time}><td>{formatForecastTime(entry.issue_time)}</td><td>{labels[entry.attention]}</td><td>{entry.warning_intervals}</td><td>{entry.removed_warning_samples}</td><td>{entry.missing_samples}</td></tr>)}</tbody></table></div><p>Revisions compare matching forecast times. Missing data does not cancel a known warning.</p></section>
    {showFuture && <details className={styles.results}><summary>Full replay evaluation · Includes future outcomes and later issues</summary>
      <label>Evaluation threshold <select disabled={!interactive} value={threshold} onChange={event => setThreshold(event.target.value)}><option value="2">Review ≥2×</option><option value="4">High ≥4×</option></select></label>
      <h2>Density errors for the selected variant</h2>
      <div className={styles.scroll}><table><thead><tr><th>Lead</th><th>Paired samples</th><th>MAPE</th><th>Bias</th><th>Underestimated by &gt;30%</th></tr></thead><tbody>{scenario.density_metrics.filter(metric => metric.variant === variant).map(metric => <tr key={metric.lead_hours}><td>+{metric.lead_hours} h</td><td>{metric.n}</td><td>{metric.mape_pct == null ? '—' : `${metric.mape_pct.toFixed(1)}%`}</td><td>{metric.bias_pct == null ? '—' : `${metric.bias_pct.toFixed(1)}%`}</td><td>{percent(metric.underestimate_gt_30pct)}</td></tr>)}</tbody></table></div>
      <h2>Warning scores for the selected variant</h2><p>Sample exceedances against the shared reference; overlapping forecasts are correlated.</p>
      <div className={styles.scroll}><table><thead><tr><th>Lead</th><th>Usable / requested</th><th>Hits</th><th>Misses</th><th>False positives</th><th>Precision</th><th>Recall</th></tr></thead><tbody>{scores.map(score => <tr key={score.lead_hours}><td>+{score.lead_hours} h</td><td>{score.usable} / {score.requested}</td><td>{score.true_positive}</td><td>{score.false_negative}</td><td>{score.false_positive}</td><td>{percent(score.precision)}</td><td>{percent(score.recall)}</td></tr>)}</tbody></table></div>
      <h2>Observed exceedance episodes</h2><div className={styles.scroll}><table><thead><tr><th>Observed start UTC</th><th>End UTC</th><th>First warning for onset sample</th><th>Notice</th></tr></thead><tbody>{events.map(event => <tr key={`${event.threshold_ratio}-${event.start}`}><td>{formatForecastTime(event.start)}</td><td>{formatForecastTime(event.end)}</td><td>{event.onset_first_warning_issue ? formatForecastTime(event.onset_first_warning_issue) : 'Onset missed'}</td><td>{event.onset_notice_hours == null ? '—' : `${event.onset_notice_hours.toFixed(0)} h${event.onset_notice_hours < 0 ? ' (late)' : ''}`}</td></tr>)}</tbody></table></div>
      <p>Notice is measured from the first forecast warning for the onset sample to that hourly observed exceedance. Warnings for later samples do not count as onset warnings. Crossing times are approximate.</p>
    </details>}
    <details><summary>Scenario assumptions and limits</summary>{scenario.limitations.map(limit => <p key={limit}>{limit}</p>)}</details>
  </div>;
}
