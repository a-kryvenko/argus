'use client';
import Link from 'next/link';
import WindChart from '../../../components/WindChart';
import { useResource } from '../../../hooks/useResource';
import { formatForecastTime } from '../../../forecast/format';
import styles from '../../../components/forecast.module.css';

type SpeedPoint = { valid_time: string; speed_background: number; speed_nominal: number; speed_early: number; speed_late: number; scenario_lower: number; scenario_upper: number };
type Event = { event_id: string; kind: string; status: string; source: string; arrival_at: string | null; source_url: string | null };
type SpeedEvent = { event_id: string; arrival_early: string; arrival_late: string; peak_speed: number; peak_time: string };
type EventForecast = { issue_time: string; events: Event[]; source_status: Record<string, { state: string }>; cme_speed?: { state: string; points: SpeedPoint[]; events: SpeedEvent[]; excluded_events: { event_id: string; reason: string }[] } | null };
const number = (v: number) => Math.round(v).toLocaleString('en-US');

export default function SolarEvents() {
  const { data, error, retry } = useResource<EventForecast>('/forecasts/solar-wind-events');
  const speed = data?.cme_speed;
  const points = speed?.points ?? [];
  const chartData = points.map(p => ({
    time: p.valid_time, median: p.speed_background, low: null, high: null,
    cmeNominal: p.speed_nominal, cmeEarly: p.speed_early, cmeLate: p.speed_late,
    cmeRange: [p.scenario_lower, p.scenario_upper] as [number, number],
  }));
  const impacts = data?.events.filter(e => e.source === 'DONKI/WSA-ENLIL' && ['predicted_hit', 'prediction_elapsed'].includes(e.status)) ?? [];
  return <main id="forecast-content" className={styles.page}>
    <div className={styles.pageHeading}><div><h1>Solar events</h1><p>Expected CME arrivals at Earth and the solar-wind speed forecast for those impacts.</p></div></div>
    {error && <p role="alert">Forecast unavailable. <button onClick={retry}>Try again</button></p>}
    {!data && !error && <p role="status">Loading the published solar-event forecast…</p>}
    {data && <>
      <p>Issued <time dateTime={data.issue_time}>{formatForecastTime(data.issue_time)}</time> · All times UTC</p>
      <p>ENLIL source: {data.source_status.WSAEnlilSimulations?.state ?? 'unavailable'}. HSS forecasts are not included.</p>
      <WindChart data={chartData} title="Speed at Earth" scenarios />
      {speed?.state === 'partial' && <p>Some arrival predictions do not have a supported speed estimate.</p>}
      <section aria-label="Predicted CME arrivals"><h2>CME arrivals</h2>
        {impacts.length === 0 ? <p>No CME impact predictions are present in this release. Check source availability above.</p> : <div className={styles.tableScroll}><table><thead><tr><th scope="col">Event</th><th scope="col">ENLIL arrival · UTC</th><th scope="col">Arrival window · UTC</th><th scope="col">Expected peak · km/s</th></tr></thead><tbody>{impacts.map(e => {
          const impact = speed?.events.find(p => p.event_id === e.event_id);
          return <tr key={e.event_id}><th scope="row">{e.source_url?.startsWith('https://') ? <a href={e.source_url} target="_blank" rel="noreferrer">{e.event_id}</a> : e.event_id}</th><td>{e.arrival_at ? formatForecastTime(e.arrival_at) : 'Unavailable'}</td><td>{impact ? `${formatForecastTime(impact.arrival_early)} – ${formatForecastTime(impact.arrival_late)}` : 'Unavailable'}</td><td>{impact ? number(impact.peak_speed) : 'Unavailable'}</td></tr>;
        })}</tbody></table></div>}
      </section>
      {points.length > 0 && <details><summary>Hourly speed forecast · km/s</summary><div className={styles.tableScroll}><table><thead><tr><th scope="col">Time · UTC</th><th scope="col">Background</th><th scope="col">Expected impact</th><th scope="col">Early</th><th scope="col">Late</th><th scope="col">Scenario range</th></tr></thead><tbody>{points.map(p => <tr key={p.valid_time}><th scope="row">{formatForecastTime(p.valid_time)}</th><td>{number(p.speed_background)}</td><td>{number(p.speed_nominal)}</td><td>{number(p.speed_early)}</td><td>{number(p.speed_late)}</td><td>{number(p.scenario_lower)}–{number(p.scenario_upper)}</td></tr>)}</tbody></table></div></details>}
    </>}
    <div className={styles.pageFooter}><Link href="/forecast/solar-wind-speed">Background speed forecast</Link></div>
  </main>;
}
