import { useId, useState } from 'react';
import type { FastBtEvaluation } from '../_lib/types';
import MetricValue from './MetricValue';
import styles from '../../../components/forecast.module.css';
import local from './metrics.module.css';

const percent = new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 1 });
const confidence = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3, signDisplay: 'exceptZero' });

export default function FastBtAccuracy({ evaluation }: { evaluation: FastBtEvaluation }) {
  const [cohortId, setCohortId] = useState('all_triggers');
  const [delay, setDelay] = useState(15);
  const id = useId();
  const cohort = evaluation.cohorts.find(cohort => cohort.id === cohortId) ?? evaluation.cohorts[0];
  const row = cohort?.by_delay.find(item => item.delay_minutes === delay) ?? cohort?.by_delay[0];
  if (!cohort || !row) return null;

  return <section className={local.fastEvaluation} aria-labelledby={`${id}-heading`}>
    <h2 id={`${id}-heading`}>Fast L1 Bt archive evaluation</h2>
    <p>{evaluation.period} · {evaluation.sample}</p>
    <div className={local.selector}>
      <label htmlFor={`${id}-cohort`}>Evaluation sample</label>
      <select id={`${id}-cohort`} value={cohort.id} onChange={event => setCohortId(event.target.value)}>
        {evaluation.cohorts.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select>
    </div>
    <div className={local.selector}>
      <label htmlFor={`${id}-delay`}>Simulated L1 data delay</label>
      <select id={`${id}-delay`} value={row.delay_minutes} onChange={event => setDelay(Number(event.target.value))}>
        {cohort.by_delay.map(item => <option key={item.delay_minutes} value={item.delay_minutes}>{item.delay_minutes} min</option>)}
      </select>
    </div>
    <p>{row.targets} matched one-hour intervals · {row.updates} with fast forecasts</p>
    <div className={styles.tableScroll}>
      <table className={local.monthlyTable}>
        <caption>Hourly forecast → Fast L1</caption>
        <thead><tr><th scope="col">Bt threshold</th><th scope="col">Observed</th><th scope="col">Brier score</th><th scope="col">Average precision</th><th scope="col">Hits</th><th scope="col">False alerts</th></tr></thead>
        <tbody>{row.thresholds.map(({ threshold, observed_events, baseline, candidate }) => <tr key={threshold}>
          <th scope="row">≥ {threshold} nT</th>
          <td>{observed_events}</td>
          <td><MetricValue value={baseline.brier_score} /> → <MetricValue value={candidate.brier_score} /></td>
          <td><MetricValue value={baseline.average_precision} /> → <MetricValue value={candidate.average_precision} /></td>
          <td>{baseline.hits} → {candidate.hits}</td>
          <td>{baseline.false_alerts} → {candidate.false_alerts}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <details>
      <summary>Evaluation notes</summary>
      {evaluation.notes.map(note => <p key={note}>{note}</p>)}
      {row.thresholds.map(({ threshold, baseline, candidate, brier_delta_ci95 }) => <p key={threshold}>
        Bt ≥ {threshold} nT · Alert cutoff {percent.format(baseline.alert_threshold)} → {percent.format(candidate.alert_threshold)}
        {brier_delta_ci95 && ` · Brier change, fast L1 minus hourly (95% interval) ${brier_delta_ci95.map(value => confidence.format(value)).join(' to ')}`}
      </p>)}
    </details>
  </section>;
}
