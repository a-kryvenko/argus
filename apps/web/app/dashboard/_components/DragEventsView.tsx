'use client';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { apiRequest } from '../../_utils/api';
import { EmptyState, Message, Pending, selectClass } from './presentation';
import { number, utc } from './leo-drag';
import type { DragEventPage, DragEvent, MonitorItem } from './drag-monitor';

const labels: Record<DragEvent['kind'], string> = {
  risk_elevated: 'Review limit reached', risk_high: 'High limit reached', risk_reduced: 'Impact reduced',
  risk_recovered: 'Below user limits', loss_increased: 'Estimated loss increased', data_stale: 'Data expired',
  assessment_unavailable: 'Assessment unavailable', data_recovered: 'Data restored',
};
const metrics = [
  ['estimated_loss_m', 'Estimated loss', 'm'], ['previous_loss_m', 'Previous loss', 'm'],
  ['change_m', 'Change', 'm'], ['change_percent', 'Change', '%'],
  ['review_limit_m', 'Review limit', 'm'], ['high_limit_m', 'High limit', 'm'], ['horizon_hours', 'Horizon', 'h'],
] as const;

export default function DragEventsView({ items, refresh, onOpen }: {
  items: MonitorItem[]; refresh: number; onOpen: (profileId: string, runId: string) => Promise<void>;
}) {
  const [data, setData] = useState<DragEventPage | null>(null);
  const [profile, setProfile] = useState('');
  const [unread, setUnread] = useState(false);
  const [page, setPage] = useState(1);
  const [version, setVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    const query = new URLSearchParams({ page: String(page), unacknowledged: String(unread) });
    if (profile) query.set('profile_id', profile);
    apiRequest<DragEventPage>(`/dashboard/drag-monitor/events?${query}`, { signal: controller.signal, cache: 'no-store' })
      .then(result => { if (!controller.signal.aborted) { setData(result); setError(''); } })
      .catch(e => { if (!controller.signal.aborted) { setData(null); setError(e instanceof Error ? e.message : 'Event journal unavailable.'); } });
    return () => controller.abort();
  }, [page, profile, unread, version, refresh]);
  async function action(operation: () => Promise<unknown>) {
    setBusy(true); setError('');
    try { await operation(); }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not complete the action.'); }
    finally { setBusy(false); }
  }
  return <Card className="min-w-0 space-y-4 p-5">
    <h2 className="font-medium">Event journal</h2>
    <p className="text-sm text-muted-foreground">Events describe changes in modeled impact and data availability. Reviewing an event records that you saw it; it does not change the spacecraft status.</p>
    <div className="flex flex-wrap items-center gap-3">
      <label className="space-y-1 text-xs">Spacecraft<select aria-label="Event spacecraft" className={selectClass} value={profile} onChange={e => { setProfile(e.target.value); setPage(1); setData(null); }}><option value="">All spacecraft</option>{items.map(item => <option key={item.profile.id} value={item.profile.id}>{item.profile.name}</option>)}</select></label>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={unread} onChange={e => { setUnread(e.target.checked); setPage(1); setData(null); }} />Unreviewed only</label>
      {data && <p className="text-sm text-muted-foreground">{data.unacknowledged} unreviewed · {data.total} {data.total === 1 ? 'event' : 'events'}</p>}
    </div>
    {error && <Message>{error}</Message>}
    {!data && !error && <Pending label="Loading events…" />}
    {data?.items.length === 0 && <EmptyState title="No events" description="Events appear when assessment or data states change." />}
    {data?.items.map(event => <article key={event.id} aria-label={`${event.spacecraft_name}: ${labels[event.kind]}`} className="space-y-3 rounded-lg border p-4 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2"><h3 className={event.severity === 'high' ? 'font-medium text-red-400' : event.severity === 'warning' ? 'font-medium text-amber-300' : 'font-medium'}>{labels[event.kind]}</h3><p className="text-xs text-muted-foreground">Detected {utc(event.detected_at)} UTC</p></div>
      <p className="font-medium">{event.spacecraft_name} · {event.active_mode} · Profile v{event.profile_revision}</p>
      <p>{event.message}</p>
      {event.values.reason === 'outside_model_domain' && <p className="text-amber-300">Active mode is outside the available grid or model limits.</p>}
      {typeof event.values.issue_time === 'string' && <p>Atmosphere issued: {utc(event.values.issue_time)} UTC</p>}
      {event.expected_at && <p>First estimated crossing: {utc(event.expected_at)} UTC</p>}
      <div className="flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted-foreground">{metrics.map(([key, label, unit]) => typeof event.values[key] === 'number' ? <span key={key}>{label}: {number(event.values[key] as number)} {unit}</span> : null)}</div>
      {event.kind === 'loss_increased' && <p className="text-xs text-muted-foreground">Growth rule: {String(event.values.growth_percent)}% and {String(event.values.growth_min_m)} m.</p>}
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="outline" disabled={busy} onClick={() => void action(() => onOpen(event.profile_id, event.run_id))}>Open assessment</Button>
        {event.comparison_run_id && <Button size="sm" variant="outline" disabled={busy} onClick={() => void action(() => onOpen(event.profile_id, event.comparison_run_id!))}>Open comparison</Button>}
        <Button size="sm" variant="outline" disabled={busy || event.acknowledged_at !== null} onClick={() => void action(async () => {
          await apiRequest(`/dashboard/drag-monitor/events/${event.id}/ack`, { method: 'POST' }); setPage(1); setVersion(v => v + 1);
        })}>{event.acknowledged_at ? 'Reviewed' : 'Mark reviewed'}</Button>
        {event.acknowledged_at && <span className="self-center text-xs text-muted-foreground">{utc(event.acknowledged_at)} UTC</span>}
      </div>
    </article>)}
    {data && <div className="flex items-center gap-3"><Button size="sm" variant="outline" disabled={page === 1 || busy} onClick={() => { setPage(page - 1); setData(null); }}>Previous events</Button><span className="text-xs">Page {page}</span><Button size="sm" variant="outline" disabled={page*20 >= data.total || busy} onClick={() => { setPage(page + 1); setData(null); }}>Next events</Button></div>}
  </Card>;
}
