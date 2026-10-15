'use client';
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { apiRequest } from '../../_utils/api';
import { EmptyState, Message, PageHeading, Pending } from './presentation';
import { number, utc } from './leo-drag';
import { attentionLabels, dataLabels, type MonitorItem, type Profile, type ProfileInput, type Run, type OperatorChart } from './drag-monitor';
import DragProfileEditor from './DragProfileEditor';
import DragOperatorPanel from './DragOperatorPanel';
import DragEventsView from './DragEventsView';

const endpoint = '/dashboard/drag-monitor';
const errorText = (error: unknown) => error instanceof Error ? error.message : 'Drag monitoring is unavailable.';

export default function DragMonitorView() {
  const [items, setItems] = useState<MonitorItem[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [editor, setEditor] = useState<Profile | null | undefined>(undefined);
  const [selected, setSelected] = useState<string | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const [pollTick, setPollTick] = useState(0);
  const [chart, setChart] = useState<OperatorChart | null>(null);
  const [chartError, setChartError] = useState('');
  const [navigation, setNavigation] = useState(0);
  const detailsRef = useRef<HTMLDivElement>(null);
  const run = runs.find(item => item.id === runId) ?? runs[0];
  const visibleRunId = run?.id;

  useEffect(() => {
    if (navigation === 0) return;
    detailsRef.current?.focus({ preventScroll: true });
    detailsRef.current?.scrollIntoView({ block: 'start' });
  }, [navigation]);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const data = await apiRequest<MonitorItem[]>(endpoint, { signal: controller.signal, cache: 'no-store' });
        if (!controller.signal.aborted) { setItems(data); setLoaded(true); setError(''); setPollTick(tick => tick + 1); }
      } catch (e) { if (!controller.signal.aborted) { setError(errorText(e)); setItems([]); setLoaded(true); } }
    }
    void load();
    const timer = setInterval(() => void load(), 60000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [version]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    apiRequest<Run[]>(`${endpoint}/${selected}/history`, { signal: controller.signal, cache: 'no-store' })
      .then(data => { if (!controller.signal.aborted) setRuns(current => {
        const selectedRun = current.find(item => item.id === runId);
        return selectedRun && !data.some(item => item.id === selectedRun.id) ? [selectedRun, ...data] : data;
      }); })
      .catch(e => { if (!controller.signal.aborted) setError(errorText(e)); });
    return () => controller.abort();
  }, [selected, version, pollTick, runId]);

  useEffect(() => {
    if (!selected || !visibleRunId) return;
    const controller = new AbortController();
    apiRequest<OperatorChart>(`${endpoint}/${selected}/chart?run_id=${encodeURIComponent(visibleRunId)}`, { signal: controller.signal, cache: 'no-store' })
      .then(data => { if (!controller.signal.aborted) { setChart(data); setChartError(''); } })
      .catch(e => { if (!controller.signal.aborted) { setChart(null); setChartError(errorText(e)); } });
    return () => controller.abort();
  }, [selected, visibleRunId, version, pollTick]);

  async function action(operation: () => Promise<unknown>) {
    setBusy(true); setError('');
    try { await operation(); setVersion(v => v + 1); }
    catch (e) { setError(errorText(e)); }
    finally { setBusy(false); }
  }
  async function save(body: ProfileInput) {
    await action(async () => {
      await apiRequest(editor ? `${endpoint}/${editor.id}?revision=${editor.revision}` : endpoint,
        { method: editor ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      setEditor(undefined);
    });
  }
  function pause(profile: Profile) {
    const { id, revision, updated_at: _updated, ...body } = profile;
    void _updated;
    void action(() => apiRequest(`${endpoint}/${id}?revision=${revision}`, { method: 'PUT',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, enabled: !profile.enabled }) }));
  }
  async function openEvent(profileId: string, eventRunId: string) {
    const saved = await apiRequest<Run>(`${endpoint}/${profileId}/runs/${eventRunId}`, { cache: 'no-store' });
    setRuns([saved]); setSelected(profileId); setRunId(eventRunId); setChart(null); setChartError('');
    setNavigation(value => value + 1);
  }
  const selectedProfile = items.find(item => item.profile.id === selected)?.profile;
  const active = run?.modes.find(mode => mode.mode === run.profile_snapshot.active_mode)?.assessment;

  return <>
    <PageHeading title="Drag risk" description="Monitor atmospheric drag against your spacecraft limits."
      action={<Button disabled={busy} onClick={() => setEditor(null)}>Add spacecraft</Button>} />
    <Card className="space-y-2 p-5 text-sm">
      <p className="font-medium">Current-condition scenarios · 24 / 48 hours</p>
      <p className="text-muted-foreground">Observed atmospheric drivers are held constant; future storms are not predicted. Each attitude mode is a separate constant-area scenario. Monitoring recalculates automatically; no maneuvers or orbit-position errors are modeled.</p>
    </Card>
    {error && <Message>{error}</Message>}
    {editor !== undefined && <DragProfileEditor key={editor ? `${editor.id}-${editor.revision}` : 'new'} profile={editor} busy={busy} onSave={save} onCancel={() => setEditor(undefined)} />}
    {!loaded && <Pending label="Loading spacecraft…" />}
    {loaded && items.length === 0 && !error && <Card><EmptyState title="No spacecraft yet" description="Add an orbit, spacecraft parameters, attitude modes and your altitude-loss limits to start monitoring." /></Card>}
    {items.length > 0 && <Card className="min-w-0 overflow-hidden"><Table>
      <TableHeader><TableRow><TableHead>Spacecraft</TableHead><TableHead>Attention</TableHead><TableHead>Data</TableHead><TableHead>Altitude loss</TableHead><TableHead>Change</TableHead><TableHead>Actions</TableHead></TableRow></TableHeader>
      <TableBody>{items.map(item => {
        const { profile } = item;
        const assessment = item.latest?.modes.find(mode => mode.mode === profile.active_mode)?.assessment;
        return <TableRow key={profile.id}>
          <TableCell><p className="font-medium">{profile.name}</p><p className="text-xs text-muted-foreground">{profile.altitude_km} km · {profile.horizon_hours} h · {profile.active_mode}{!profile.enabled && ' · Paused'}</p></TableCell>
          <TableCell><span className={item.attention === 'high' ? 'text-red-400' : item.attention === 'elevated' ? 'text-amber-400' : ''}>{attentionLabels[item.attention]}</span></TableCell>
          <TableCell>{dataLabels[item.data_status]}</TableCell>
          <TableCell>{item.data_status === 'current' && assessment ? `${number(assessment.estimated_altitude_loss_m)} m` : '—'}</TableCell>
          <TableCell>{item.loss_change_m === null ? '—' : `${item.loss_change_m > 0 ? '+' : ''}${number(item.loss_change_m)} m`}</TableCell>
          <TableCell><div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" disabled={busy} onClick={() => { setRuns([]); setRunId(null); setChart(null); setChartError(''); setSelected(profile.id); setVersion(v => v + 1); }}>Details</Button>
            <Button size="sm" variant="outline" disabled={busy} onClick={() => setEditor(profile)}>Edit</Button>
            <Button size="sm" variant="outline" disabled={busy || !profile.enabled} onClick={() => void action(() => apiRequest(`${endpoint}/${profile.id}/assess`, { method: 'POST' }))}>Assess now</Button>
            <Button size="sm" variant="outline" disabled={busy} onClick={() => pause(profile)}>{profile.enabled ? 'Pause' : 'Resume'}</Button>
          </div></TableCell>
        </TableRow>;
      })}</TableBody>
    </Table><p className="px-4 pb-4 text-xs text-muted-foreground">Change compares different atmosphere releases with identical profile parameters and density model. Data status concerns atmosphere freshness; orbit inputs are supplied manually.</p></Card>}
    <DragEventsView items={items} refresh={pollTick} onOpen={openEvent} />
    {selected && <Card ref={detailsRef} tabIndex={-1} className="min-w-0 space-y-4 p-5">
      <h2 className="font-medium">{selectedProfile?.name} · Assessment history</h2>
      <p className="text-xs text-muted-foreground">Saved snapshots retain their original parameters and thresholds. Crossing times are the first hourly samples at or above a limit.</p>
      {runs.length === 0 && <p className="text-sm text-muted-foreground">No saved assessments loaded.</p>}
      <div className="flex flex-wrap gap-2">{runs.map(entry => <Button key={entry.id} size="sm" variant={run?.id === entry.id ? 'default' : 'outline'} onClick={() => { setRunId(entry.id); setChart(null); setChartError(''); }}>{utc(entry.assessed_at)} · v{entry.profile_revision}</Button>)}</div>
      {run && <>
        <p className="text-sm">Saved assessment · Profile v{run.profile_revision} · {run.profile_snapshot.horizon_hours} h · Active mode: {run.profile_snapshot.active_mode}</p>
        {chartError && <Message>{chartError}</Message>}
        {chart?.run_id === run.id ? <DragOperatorPanel chart={chart} /> : !chartError && <Pending label="Preparing operator chart…" />}
        {run.modes.map(mode => <div key={mode.mode} className="space-y-2 rounded-lg border p-4 text-sm">
          <h3 className="font-medium">{mode.mode}{mode.mode === run.profile_snapshot.active_mode && ' · Active'}</h3>
          {mode.assessment ? <>
            <p>{number(mode.assessment.estimated_altitude_loss_m)} m altitude loss · {number(mode.assessment.delta_v_loss_m_s)} m/s drag Δv</p>
            <p>{mode.assessment.risk_reason}</p>
            <p>Review limit: {mode.elevated_at ? `${utc(mode.elevated_at)} UTC` : 'Not reached'} · High limit: {mode.high_at ? `${utc(mode.high_at)} UTC` : 'Not reached'}</p>
            <p className="text-xs text-muted-foreground">{utc(mode.assessment.start_time)} – {utc(mode.assessment.end_time)} UTC · {mode.assessment.source.density_model}</p>
          </> : <p>{mode.error === 'outside_model_domain' ? 'Outside the density grid or fixed-orbit model limits (maximum estimated loss 1,000 m).' : 'Atmosphere or calculation service unavailable. No risk category assigned.'}</p>}
        </div>)}
        {active && <>
          <details className="text-sm"><summary className="cursor-pointer">Hourly estimates</summary>
            <Table className="mt-3"><TableHeader><TableRow><TableHead>Time (UTC)</TableHead><TableHead>Density (kg/m³)</TableHead><TableHead>Drag Δv (m/s)</TableHead><TableHead>Altitude loss (m)</TableHead></TableRow></TableHeader>
              <TableBody>{active.predictions.map(point => <TableRow key={point.lead_hours}><TableCell className="whitespace-nowrap">{utc(point.valid_time)}</TableCell><TableCell>{number(point.mean_density_kg_m3)}</TableCell><TableCell>{number(point.delta_v_loss_m_s)}</TableCell><TableCell>{number(point.estimated_altitude_loss_m)}</TableCell></TableRow>)}</TableBody>
            </Table>
          </details>
          <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">Source and assumptions</summary>
          <p className="mt-3">Release: {active.source.release_id} · Solar observations: {utc(active.source.observed_at)} UTC · Geomagnetic observations: {active.source.geomagnetic_observed_at ? utc(active.source.geomagnetic_observed_at) : 'Unavailable'} UTC</p>
          <ul className="my-3 list-disc space-y-1 pl-5">{active.assumptions.map(item => <li key={item}>{item}</li>)}</ul>
        </details></>}
        <details className="text-xs"><summary className="cursor-pointer">Saved spacecraft parameters and limits</summary>
          <div className="mt-3 space-y-2">
            <p>{run.profile_snapshot.name} · {run.profile_snapshot.altitude_km} km · {run.profile_snapshot.inclination_deg}° · {run.profile_snapshot.mass_kg} kg</p>
            <p>Review: {run.profile_snapshot.thresholds.elevated_altitude_loss_m} m · High: {run.profile_snapshot.thresholds.high_altitude_loss_m} m over {run.profile_snapshot.horizon_hours} h</p>
            {run.profile_snapshot.modes.map(mode => <p key={mode.name}>{mode.name}: {mode.effective_area_m2} m² · Cd {mode.drag_coefficient}</p>)}
          </div>
        </details>
      </>}
    </Card>}
  </>;
}
