'use client';
import WindChart from '../../_components/WindChart';
import { formatForecastTime } from '../../_utils/forecast';
import type { DemoEventReport } from '../_lib/replay';
import styles from '../../_components/forecast.module.css';

export default function DemoCME({ report, compact = false }: { report: DemoEventReport | null; compact?: boolean }) {
  if (!report) return <p role="status">No historical ENLIL release is available for this hour.</p>;
  const impacts = report.events.filter(event => event.arrival_at && ['predicted_hit', 'prediction_elapsed'].includes(event.status));
  const speed = report.cme_speed;
  const rows = (speed?.points ?? []).map(p => ({ time: p.valid_time, median: p.speed_background, low: null, high: null,
    cmeNominal: p.speed_nominal, cmeEarly: p.speed_early, cmeLate: p.speed_late,
    cmeRange: [p.scenario_lower, p.scenario_upper] as [number, number] }));
  return <section aria-label="Historical CME arrival context">
    <h2>ENLIL CME arrivals · Earth</h2>
    <p>Issued {formatForecastTime(report.issue_time)} · 168 h arrival context. Availability follows archived NASA/DONKI notification publication; Argus receipt times are unknown.</p>
    {impacts.length ? <div className={styles.tableScroll}><table><thead><tr><th>CME</th><th>Published UTC</th><th>ENLIL arrival UTC</th><th>Status</th></tr></thead><tbody>{impacts.map(event => <tr key={event.event_id}><td>{event.event_id}</td><td>{formatForecastTime(event.available_at)}</td><td>{formatForecastTime(event.arrival_at!)}</td><td>{event.status === 'prediction_elapsed' ? 'Predicted time elapsed' : 'Predicted impact'}</td></tr>)}</tbody></table></div> : <p>No published impact prediction in the supported single-CME archive at this hour. This does not establish absence of incoming CMEs.</p>}
    <p>Single-CME notifications only; ambiguous multi-CME notices are excluded. Arrival spreads and speed scenarios are uncalibrated.</p>
    {compact ? <p>CME context is a separate watch. It does not multiply density or predict storm strength.</p> : <>
      {!!speed?.events.length && <WindChart title="CME speed scenarios" data={rows} scenarios />}
      {speed?.state === 'unavailable' && <p>CME speed scenarios are unavailable for this issue.</p>}
      {speed?.state === 'partial' && <p>Some published arrivals have no supported speed estimate.</p>}
    </>}
  </section>;
}
