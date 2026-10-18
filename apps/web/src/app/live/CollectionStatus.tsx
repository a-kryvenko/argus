'use client';
import { useLiveQuery } from './useLiveQuery';
import styles from './page.module.css';

type SourceStatus = {
  source_id: string;
  label: string;
  source_url: string;
  status: 'ok' | 'not_started' | 'collector_stalled' | 'collector_overdue' | 'collection_error' | 'collecting' | 'source_delayed' | 'data_unavailable' | 'data_partial';
  poll_seconds: number;
  last_attempt_at: string | null;
  last_completed_at: string | null;
  last_response_at: string | null;
  last_success_at: string | null;
  last_error_at: string | null;
  last_error_message: string | null;
  last_error_code: string | null;
  consecutive_failures: number;
  latest_observation_at: string | null;
  latest_interval_end: string | null;
};
type Status = { generated_at: string; status: 'ok' | 'degraded'; sources: Record<string, SourceStatus> };
const descriptions: Record<SourceStatus['status'], [string, string]> = {
  ok: ['Updating', 'Collection is running and the source data is within its freshness window.'],
  not_started: ['Not monitored yet', 'No collection attempt has been recorded.'],
  collector_stalled: ['Attempt unfinished', 'The last attempt did not finish in time. Collection may be stalled or interrupted.'],
  collector_overdue: ['Collection overdue', 'No recent collection attempt. The collector may be stopped or stalled.'],
  collection_error: ['Collection error', 'Recent attempts failed. The last error is shown below.'],
  collecting: ['Fetching data', 'An initial collection attempt is in progress.'],
  source_delayed: ['Source data delayed', 'The source responds, but its latest measurements are outside the freshness window.'],
  data_unavailable: ['Source data unavailable', 'The source responds, but its latest records have no usable measurements.'],
  data_partial: ['Incomplete source data', 'Some latest measurements are missing or flagged.'],
};
function timestamp(value: string | null) {
  return value ? new Date(value).toISOString().replace('T', ' ').slice(0,19) + ' UTC' : 'Not recorded';
}

export default function CollectionStatus({ now }: { now?: number }) {
  const { data, failed, receivedAt } = useLiveQuery<Status>('/public/observations/status');
  const old = receivedAt && now ? now-receivedAt > 120000 : false;
  const sources = Object.values(data?.sources ?? {});
  const affected = sources.filter(source => source.status !== 'ok');
  const label = failed ? 'Status could not be refreshed' : old ? 'Status update overdue' : !data ? 'Checking sources…' : affected.length ? `${affected.length} of ${sources.length} sources need attention` : `${sources.length} sources updating`;
  return <details className={styles.collectionStatus}>
    <summary>Data collection · <span className={failed || old || affected.length ? styles.warning : undefined}>{label}</span></summary>
    {(failed || old) && <p role="alert" className={styles.warning}>Current collection status is unavailable. {data && 'Showing the last received status below.'} Retrying every minute.</p>}
    {data && <p className={styles.sampleTime}>Status as of {timestamp(data.generated_at)}</p>}
    <div className={styles.collectionGrid}>
      {sources.map(source => <section key={source.source_id}>
        <h3>{source.source_id === 'kp' ? 'Kp' : source.label}</h3>
        {source.source_id === 'kp' && <p className={styles.sampleTime}>NOAA SWPC · Preliminary estimate</p>}
        <p className={source.status === 'ok' ? undefined : styles.warning}>{descriptions[source.status][0]}</p>
        <p>{descriptions[source.status][1]}</p>
        <dl>
          <dt>Collection schedule</dt><dd>Every {source.poll_seconds/60} min</dd>
          <dt>Last attempt</dt><dd>{timestamp(source.last_attempt_at)}</dd>
          <dt>Last attempt completed</dt><dd>{timestamp(source.last_completed_at)}</dd>
          <dt>Last valid response</dt><dd>{timestamp(source.last_response_at)}</dd>
          <dt>Last successful save</dt><dd>{timestamp(source.last_success_at)}</dd>
          <dt>Latest source measurement</dt><dd>{timestamp(source.latest_observation_at)}</dd>
          {source.latest_interval_end && <><dt>Measurement interval ends</dt><dd>{timestamp(source.latest_interval_end)}</dd></>}
          <dt>Consecutive failures</dt><dd>{source.consecutive_failures}</dd>
          {source.last_error_at && <><dt>{source.consecutive_failures ? 'Last error' : 'Last error (recovered)'}</dt><dd>{source.last_error_message}<br />{timestamp(source.last_error_at)}</dd></>}
        </dl>
        <a href={source.source_url}>Source data</a>
      </section>)}
    </div>
    <a href="/api/v1/public/observations/status">Collection status · JSON</a>
  </details>;
}
