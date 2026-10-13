'use client';
import Link from 'next/link';
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useResource } from '../../_utils/useResource';
import { formatForecastTime } from '../../_utils/forecast';
import styles from '../../_components/forecast.module.css';

type SpeedPoint = { valid_time: string; speed_background: number; speed_nominal: number; speed_early: number; speed_late: number; scenario_lower: number; scenario_upper: number };
type Event = { event_id: string; kind: string; status: string; source: string; arrival_at: string | null; source_url: string | null };
type SpeedEvent = { event_id: string; arrival_early: string; arrival_late: string; peak_speed: number; peak_time: string };
type EventForecast = { issue_time: string; events: Event[]; source_status: Record<string, { state: string }>; cme_speed?: { state: string; points: SpeedPoint[]; events: SpeedEvent[]; excluded_events: { event_id: string; reason: string }[] } | null };
const number = (v: number) => Math.round(v).toLocaleString('en-US');

export default function SolarEvents() {
  const { data, error, retry } = useResource<EventForecast>('/public/forecasts/solar-wind-events');
  const speed = data?.cme_speed;
  const points = (speed?.points ?? []).map(p => ({ ...p, timestamp: Date.parse(p.valid_time), band: [p.scenario_lower, p.scenario_upper] }));
  const impacts = data?.events.filter(e => e.source === 'DONKI/WSA-ENLIL' && ['predicted_hit', 'prediction_elapsed'].includes(e.status)) ?? [];
  return <main id="forecast-content" className={styles.page}>
    <div className={styles.pageHeading}><div><div className={styles.eyebrow}>SOLAR EVENTS · CME</div><h1>Solar events</h1><p>Expected CME arrivals at Earth and the solar-wind speed forecast for those impacts.</p></div><button onClick={retry}>Refresh forecast</button></div>
    {error && <p role="alert">Forecast unavailable. <button onClick={retry}>Try again</button></p>}
    {!data && !error && <p role="status">Loading the published solar-event forecast…</p>}
    {data && <>
      <p>Issued <time dateTime={data.issue_time}>{formatForecastTime(data.issue_time)}</time> · All times UTC</p>
      <p>ENLIL source: {data.source_status.WSAEnlilSimulations?.state ?? 'unavailable'}. HSS forecasts are not included.</p>
      <section className={styles.chart} aria-label="CME-conditioned solar-wind speed forecast">
        <div className={styles.chartHeading}><h2>Speed at Earth</h2><span>km/s</span></div>
        <div className={styles.chartLegend}><span>━ Background</span><span style={{ color: '#f1c777' }}>━ Expected CME impact</span><span style={{ color: '#73c9b0' }}>┄ Early arrival</span><span style={{ color: '#bf9bf1' }}>┄ Late arrival</span></div>
        {points.length > 0 ? <div className={styles.quantilePlot}><ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 12, right: 35, bottom: 8, left: 0 }}>
            <CartesianGrid vertical={false} stroke="#26333f" />
            <XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} minTickGap={50} tick={{ fill: '#92a4b5', fontSize: 10 }} tickFormatter={v => new Date(v).toISOString().slice(5,16).replace('T',' ')} />
            <YAxis width={62} domain={['auto','auto']} tick={{ fill: '#92a4b5', fontSize: 10 }} />
            <Tooltip labelFormatter={v => formatForecastTime(new Date(Number(v)).toISOString())} />
            <Area dataKey="band" name="Possible scenario envelope (km/s)" fill="#65b5eb" fillOpacity={.12} stroke="none" isAnimationActive={false} />
            <Line dataKey="speed_background" name="Background (km/s)" stroke="#92a4b5" dot={false} isAnimationActive={false} />
            <Line dataKey="speed_nominal" name="Expected impact (km/s)" stroke="#f1c777" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="speed_early" name="Early arrival (km/s)" stroke="#73c9b0" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
            <Line dataKey="speed_late" name="Late arrival (km/s)" stroke="#bf9bf1" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer></div> : <p role="status">The speed forecast is currently unavailable. Arrival predictions remain listed below.</p>}
        <p className={styles.chartNote}>The impact forecast assumes the predicted CME reaches Earth. Early and late arrivals show timing uncertainty; the shaded envelope is a range of scenarios, not a probability interval.</p>
        {speed?.state === 'partial' && <p>Some arrival predictions do not have a supported speed estimate.</p>}
      </section>
      <section aria-label="Predicted CME arrivals"><h2>CME arrivals</h2>
        {impacts.length === 0 ? <p>No CME impact predictions are present in this release. Check source availability above.</p> : <div className={styles.tableScroll}><table><thead><tr><th scope="col">Event</th><th scope="col">ENLIL arrival · UTC</th><th scope="col">Arrival window · UTC</th><th scope="col">Expected peak · km/s</th></tr></thead><tbody>{impacts.map(e => {
          const impact = speed?.events.find(p => p.event_id === e.event_id);
          return <tr key={e.event_id}><th scope="row">{e.source_url?.startsWith('https://') ? <a href={e.source_url} target="_blank" rel="noreferrer">{e.event_id}</a> : e.event_id}</th><td>{e.arrival_at ? formatForecastTime(e.arrival_at) : 'Unavailable'}</td><td>{impact ? `${formatForecastTime(impact.arrival_early)} – ${formatForecastTime(impact.arrival_late)}` : 'Unavailable'}</td><td>{impact ? number(impact.peak_speed) : 'Unavailable'}</td></tr>;
        })}</tbody></table></div>}
      </section>
      {points.length > 0 && <details><summary>Hourly speed forecast · km/s</summary><div className={styles.tableScroll}><table><thead><tr><th scope="col">Time · UTC</th><th scope="col">Background</th><th scope="col">Expected impact</th><th scope="col">Early</th><th scope="col">Late</th><th scope="col">Scenario range</th></tr></thead><tbody>{points.map(p => <tr key={p.valid_time}><th scope="row">{formatForecastTime(p.valid_time)}</th><td>{number(p.speed_background)}</td><td>{number(p.speed_nominal)}</td><td>{number(p.speed_early)}</td><td>{number(p.speed_late)}</td><td>{number(p.scenario_lower)}–{number(p.scenario_upper)}</td></tr>)}</tbody></table></div></details>}
    </>}
    <div className={styles.pageFooter}><Link href="/products">All forecast products</Link><Link href="/products/solar-wind-speed">Background speed forecast</Link></div>
  </main>;
}
