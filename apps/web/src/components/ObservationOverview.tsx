'use client';

import { useState } from 'react';
import WorkspaceShell from './WorkspaceShell';
import SDOChannels from './SDOChannels';
import { useResource } from '../hooks/useResource';
import styles from './overview.module.css';

type Risk = 0 | 1 | 2 | 3 | 4;
type Observation = {
  title: string;
  value: number | null;
  measurements: string;
  observed_at: string | null;
  risk_level: Risk;
  image_name: string | null;
};
type Group = { title: string; items: Observation[] };
type CmeEvent = {
  event_id: string;
  source_url: string | null;
  observed_at: string | null;
  arrival_window_start: string | null;
  arrival_window_end: string | null;
};
type Summary = {
  generated_at: string;
  observations: Record<'plasma' | 'geo' | 'hmf' | 'radio', Group>;
  cme_events: CmeEvent[];
  cme_events_available: boolean;
  sdo_images: Record<string, string>;
};

const groups = ['plasma', 'geo', 'hmf', 'radio'] as const;
const riskNames = ['Green', 'Yellow', 'Orange', 'Red', 'Crimson'];
const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
const date = new Intl.DateTimeFormat('en-GB', {
  month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', timeZone: 'UTC', hourCycle: 'h23',
});

function Time({ value }: { value: string | null }) {
  return value ? <time dateTime={value}>{date.format(new Date(value))}</time> : <span className={styles.muted}>—</span>;
}

function Sparkline({ item }: { item: Observation }) {
  const [failed, setFailed] = useState(false);
  if (!item.image_name || failed) return null;
  // SVGs are already generated and sized by the API; no client chart library is needed.
  // eslint-disable-next-line @next/next/no-img-element
  return <img className={styles.sparkline} src={`/api/v1/summary/images/${encodeURIComponent(item.image_name)}`}
    alt={`${item.title}: recent observations and forecast`} width={480} height={240} onError={() => setFailed(true)} />;
}

function ObservationCard({ group, id }: { group: Group; id: string }) {
  const risk = Math.max(0, ...group.items.map(item => item.risk_level)) as Risk;
  return <section className={styles.card} data-risk={risk} aria-labelledby={`${id}-title`}>
    <header className={styles.cardHeader}><h2 id={`${id}-title`}>{group.title}</h2></header>
    <ul className={styles.items}>{group.items.map(item => <li key={item.title} className={styles.item} data-risk={item.risk_level}>
      <span className={styles.variable}>{item.title}<span className={styles.srOnly}> · {riskNames[item.risk_level]} risk</span></span>
      <div className={styles.reading} title={item.observed_at ? `Observed ${date.format(new Date(item.observed_at))} UTC` : 'Observation unavailable'}>
        <span className={styles.value}>{item.value === null ? '—' : number.format(item.value)}</span>
        <span className={styles.unit}>{item.measurements}</span>
      </div>
      <Sparkline key={item.image_name} item={item} />
    </li>)}</ul>
  </section>;
}

export default function ObservationOverview() {
  const { data, error, retry } = useResource<Summary>('/summary', 5 * 60 * 1000);
  return <WorkspaceShell section="Overview" contentId="overview-content">
    <main id="overview-content" className={styles.page}>
      <header className={styles.heading}><h1>Space weather</h1>
        <a className={styles.apiLink} href="/api/v1/summary" target="_blank" rel="noreferrer" aria-label="Summary API (opens in a new tab)">API ↗</a>
      </header>
      <SDOChannels assets={data?.sdo_images ?? {}} />
      {!data && <div className={styles.state} role={error ? 'alert' : 'status'}>{error ?? 'Loading observations and CME events…'}
        {error && <button onClick={retry}>Try again</button>}
      </div>}
      {data && <>
        {data.cme_events.length > 0 && <section className={`${styles.card} ${styles.events}`} aria-labelledby="cme-title">
          <header className={styles.cardHeader}><div><h2 id="cme-title">CME events</h2><p>Last 7 days · all times UTC</p></div><span className={styles.eventCount}>{data.cme_events.length} events</span></header>
          {!data.cme_events_available ? <p className={styles.state}>CME event data is unavailable.</p>
            : <div className={styles.tableScroll} role="region" aria-label="CME events table" tabIndex={0}><table>
              <thead><tr><th scope="col">Event</th><th scope="col">Observed</th><th scope="col">Arrival window start</th><th scope="col">Arrival window end</th></tr></thead>
              <tbody>{data.cme_events.map(event => <tr key={event.event_id}>
                <th scope="row">{event.source_url && /^https?:\/\//i.test(event.source_url)
                  ? <a href={event.source_url} target="_blank" rel="noreferrer">{event.event_id}<span aria-hidden="true"> ↗</span><span className={styles.srOnly}> (source, opens in a new tab)</span></a> : event.event_id}</th>
                <td><Time value={event.observed_at} /></td>
                <td><Time value={event.arrival_window_start} /></td><td><Time value={event.arrival_window_end} /></td>
              </tr>)}</tbody>
            </table></div>}
        </section>}
        <div className={styles.grid}>{groups.map(id => <ObservationCard key={id} id={id} group={data.observations[id]} />)}</div>
      </>}
    </main>
  </WorkspaceShell>;
}
