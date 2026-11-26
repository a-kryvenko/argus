import { useId, useState } from 'react';
import type { FastL1Evaluation } from '../_lib/types';
import styles from '../../../components/forecast.module.css';
import local from './metrics.module.css';

const score = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 });
const interval = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
const perCent = new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 1 });
const confidence = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3, signDisplay: 'exceptZero' });

export default function FastL1Accuracy({ evaluation }: { evaluation: FastL1Evaluation }) {
  const [delay, setDelay] = useState(15);
  const id = useId();
  const row = evaluation.by_delay.find(item => item.delay_minutes === delay) ?? evaluation.by_delay[0];
  if (!row) return null;
  const { baseline, candidate } = row;
  const scores = [
    ['Kp ≥ 4 Brier score', score.format(baseline.kp4_brier), score.format(candidate.kp4_brier)],
    ['Kp ≥ 5 Brier score', score.format(baseline.kp5_brier), score.format(candidate.kp5_brier)],
    ['Kp ≥ 6 Brier score', score.format(baseline.kp6_brier), score.format(candidate.kp6_brier)],
    ['ap mean absolute error', interval.format(baseline.ap_mae), interval.format(candidate.ap_mae)],
    ['ap 80% interval coverage', perCent.format(baseline.ap_coverage_80), perCent.format(candidate.ap_coverage_80)],
    ['ap 80% interval width', interval.format(baseline.ap_interval_width_80), interval.format(candidate.ap_interval_width_80)],
    ['Kp ≥ 5 alert cutoff', perCent.format(baseline.alert_threshold), perCent.format(candidate.alert_threshold)],
    ['Storm intervals detected', `${baseline.hits} of ${row.observed_events}`, `${candidate.hits} of ${row.observed_events}`],
    ['False alerts', baseline.false_alerts, candidate.false_alerts],
  ];
  return <section className={local.fastEvaluation} aria-labelledby={`${id}-heading`}>
    <h2 id={`${id}-heading`}>Fast L1 archive evaluation</h2>
    <p>{evaluation.period} · {evaluation.sample}</p>
    <div className={local.selector}>
      <label htmlFor={`${id}-delay`}>Simulated L1 data delay</label>
      <select id={`${id}-delay`} value={row.delay_minutes} onChange={event => setDelay(Number(event.target.value))}>
        {evaluation.by_delay.map(item => <option key={item.delay_minutes} value={item.delay_minutes}>{item.delay_minutes} min</option>)}
      </select>
    </div>
    <p>{row.targets} matched 3-hour intervals · {row.updates} fast updates</p>
    <div className={styles.tableScroll}>
      <table className={local.monthlyTable} aria-label="Hourly and fast L1 scores">
        <thead><tr><th scope="col">Metric</th><th scope="col">Hourly forecast</th><th scope="col">Fast L1</th></tr></thead>
        <tbody>{scores.map(([label, current, fast]) => <tr key={label}>
          <th scope="row">{label}</th><td>{current}</td><td>{fast}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <details>
      <summary>Evaluation notes</summary>
      {evaluation.notes.map(note => <p key={note}>{note}</p>)}
      {row.kp5_brier_delta_ci95 && <p>Kp ≥ 5 Brier change, fast L1 minus hourly (95% interval) · {row.kp5_brier_delta_ci95.map(value => confidence.format(value)).join(' to ')}</p>}
      {row.ap_mae_delta_ci95 && <p>ap MAE change, fast L1 minus hourly (95% interval) · {row.ap_mae_delta_ci95.map(value => confidence.format(value)).join(' to ')}</p>}
    </details>
  </section>;
}
