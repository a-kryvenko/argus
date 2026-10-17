'use client';
import { useEffect, useRef, useState } from 'react';
import { CartesianGrid, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { apiRequest } from '../../_utils/api';
import { EmptyState, Message, PageHeading } from './presentation';
import { utc, spacecraftStatus, attentionLabel, statusLabel, type DensityItem, type DensityProfile, type DensityProfileInput, type DensityRun, type DensityRunSummary, type DensityEventPage } from './density-advisory';

const endpoint = '/dashboard/density-advisory';
const problem = (e: unknown) => e instanceof Error ? e.message : 'Density advisory is unavailable.';
const density = (v: number | null | undefined) => v == null ? '—' : `${v.toExponential(3)} kg/m³`;
const inputClass = 'w-full rounded-md border bg-background p-2 text-sm';

function Editor({ profile, save, cancel, busy }: { profile: DensityProfile | null; save: (body: DensityProfileInput) => Promise<void>; cancel: () => void; busy: boolean }) {
  const [name, setName] = useState(profile?.name ?? '');
  const [apoapsis, setApoapsis] = useState(profile?.apoapsis_km?.toString() ?? '');
  const [periapsis, setPeriapsis] = useState(profile?.periapsis_km?.toString() ?? '');
  const [inclination, setInclination] = useState(profile?.inclination_deg?.toString() ?? '');
  const [weight, setWeight] = useState(profile?.weight_kg?.toString() ?? '');
  const [area, setArea] = useState(profile?.effective_area_m2?.toString() ?? '');
  const [review, setReview] = useState(profile?.policy.review_ratio ?? 2);
  const [high, setHigh] = useState(profile?.policy.high_ratio ?? 4);
  const [error, setError] = useState('');
  async function submit(e: React.FormEvent) {
    e.preventDefault(); setError('');
    try {
      if (Number(apoapsis) < Number(periapsis)) throw new Error('Apoapsis must be greater than or equal to periapsis.');
      if (high <= review) throw new Error('High density ratio must exceed review density ratio.');
      if ((weight !== '' && Number(weight) <= 0) || (area !== '' && Number(area) <= 0)) throw new Error('Weight and effective area must be positive when supplied.');
      await save({ name, apoapsis_km: Number(apoapsis), periapsis_km: Number(periapsis), inclination_deg: Number(inclination),
        weight_kg: weight === '' ? null : Number(weight), effective_area_m2: area === '' ? null : Number(area),
        policy: { review_ratio: review, high_ratio: high } });
    } catch (e) { setError(problem(e)); }
  }
  return <Card className="space-y-4 p-5"><h2 className="text-lg font-semibold">{profile ? 'Update spacecraft' : 'Add spacecraft'}</h2>
    <form className="space-y-4" onSubmit={submit}>
      <label className="block">Spacecraft name<input className={inputClass} required maxLength={120} value={name} onChange={e => setName(e.target.value)} /></label>
      <div className="grid gap-3 sm:grid-cols-3">
        <label>Apoapsis (km)<input className={inputClass} type="number" required min="90" max="2500" step="any" value={apoapsis} onChange={e => setApoapsis(e.target.value)} /></label>
        <label>Periapsis (km)<input className={inputClass} type="number" required min="90" max="2500" step="any" value={periapsis} onChange={e => setPeriapsis(e.target.value)} /></label>
        <label>Inclination (°)<input className={inputClass} type="number" required min="0" max="180" step="any" value={inclination} onChange={e => setInclination(e.target.value)} /></label>
        <label>Weight (kg, optional)<input className={inputClass} type="number" min="0" step="any" value={weight} onChange={e => setWeight(e.target.value)} /></label>
        <label>Effective area (m², optional)<input className={inputClass} type="number" min="0" step="any" value={area} onChange={e => setArea(e.target.value)} /></label>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <label>Review density ratio<input className={inputClass} type="number" required min="1.01" max="100" step="any" value={review} onChange={e => setReview(Number(e.target.value))} /></label>
        <label>High density ratio<input className={inputClass} type="number" required min="1.01" max="100" step="any" value={high} onChange={e => setHigh(Number(e.target.value))} /></label>
      </div>
      {error && <Message>{error}</Message>}
      <div className="flex gap-2"><Button disabled={busy} type="submit">Save spacecraft</Button><Button type="button" variant="outline" onClick={cancel}>Cancel</Button></div>
    </form>
  </Card>;
}

function RunDetails({ run }: { run: DensityRun }) {
  const [samplePage, setSamplePage] = useState(0);
  const a = run.assessment;
  const orbit = a.source.assessment_basis === 'orbit_envelope';
  const points = a.samples.filter(p => p.ratio !== null).map(p => ({ time: Date.parse(p.time), ratio: p.ratio }));
  const watches = a.intervals.filter(w => w.attention === 'watch');
  return <Card className="min-w-0 space-y-4 p-5">
    <h2 className="text-lg font-semibold">Saved density forecast · {run.profile_snapshot.name} · v{run.profile_revision}</h2>
    <p className="text-sm">Snapshot at {utc(a.computed_at)} · {attentionLabel[a.attention]} · {statusLabel[a.data_status]}. Historical snapshot; current status is shown in the spacecraft list.</p>
    <p className="text-sm text-muted-foreground">Atmospheric forecasts hold observed drivers constant. CME arrival context is separate and does not predict atmospheric storm strength.</p>
    <dl className="grid gap-2 text-sm sm:grid-cols-2">
      <div><dt className="text-muted-foreground">Spacecraft orbit</dt><dd>Apoapsis: {run.profile_snapshot.apoapsis_km ?? '—'} km · periapsis: {run.profile_snapshot.periapsis_km ?? '—'} km · inclination: {run.profile_snapshot.inclination_deg ?? '—'}°</dd></div>
      <div><dt className="text-muted-foreground">Spacecraft properties</dt><dd>Weight: {run.profile_snapshot.weight_kg ?? '—'} kg · effective area: {run.profile_snapshot.effective_area_m2 ?? '—'} m²</dd></div>
      <div><dt className="text-muted-foreground">Density forecast</dt><dd>{a.source.density_model ?? 'Unavailable'} · issued {a.source.density_issue_time ? utc(a.source.density_issue_time) : '—'}</dd><dd>Observed drivers: {a.source.density_observed_at ? utc(a.source.density_observed_at) : '—'}</dd></div>
      <div><dt className="text-muted-foreground">CME context</dt><dd>{a.source.cme_status} · issued {a.source.cme_issue_time ? utc(a.source.cme_issue_time) : '—'}</dd></div>
      {!orbit && a.source.trajectory_generated_at && <div><dt className="text-muted-foreground">Trajectory</dt><dd>{a.source.trajectory_source} · generated {utc(a.source.trajectory_generated_at)}</dd></div>}
      <div><dt className="text-muted-foreground">Reference</dt><dd>{a.source.reference ? `${a.source.reference.label} · ${a.source.reference.source}` : 'Unavailable — relative anomaly not assessed'}</dd>{a.source.reference && <dd>{a.source.reference.kind === 'model_geomagnetic_reference' ? 'Model background' : 'Operator reference'} · prepared {utc(a.source.reference.prepared_at)}{a.source.reference.method && ` · ${a.source.reference.method}`}</dd>}</div>
    </dl>
    <div role="img" aria-label={orbit ? "Maximum forecast density ratio across the orbital envelope; CME arrival context" : "Forecast density ratio at supplied trajectory samples; CME arrival context"} className="h-72 min-w-0">
      <ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 15, right: 20, bottom: 20, left: 5 }}>
        <CartesianGrid strokeDasharray="3 3" /><XAxis type="number" dataKey="time" domain={[Date.parse(a.samples[0]?.time ?? a.computed_at), Date.parse(a.samples.at(-1)?.time ?? a.computed_at)]} tickFormatter={v => new Date(v).toISOString().slice(11, 16)} name="UTC" />
        <YAxis type="number" dataKey="ratio" name="Density / reference" /><Tooltip labelFormatter={v => utc(new Date(Number(v)).toISOString())} />
        <ReferenceLine y={run.profile_snapshot.policy.review_ratio} stroke="#f59e0b" label="Review" />
        <ReferenceLine y={run.profile_snapshot.policy.high_ratio} stroke="#f87171" label="High" />
        {watches.map((w, i) => w.start === w.end ? <ReferenceLine key={i} x={Date.parse(w.start)} stroke="#a78bfa" label="CME" /> : <ReferenceArea key={i} x1={Date.parse(w.start)} x2={Date.parse(w.end)} fill="#a78bfa" fillOpacity={0.15} />)}
        <Scatter data={points} fill="#38bdf8" isAnimationActive={false} />
      </ScatterChart></ResponsiveContainer>
    </div>
    <p className="text-xs text-muted-foreground">{orbit ? "Dots show the largest density ratio across the orbital envelope at each forecast time. They do not show spacecraft position or exposure. Purple marks show separate CME arrival context." : "Dots are forecast values at supplied samples, not observations. Purple marks show separate CME arrival context. No continuous trajectory or sub-sample crossing is inferred."}</p>
    <h3 className="font-semibold">Attention intervals and reasons</h3>
    {!a.intervals.length && <p>No attention interval identified in this snapshot. Check data coverage and reference availability.</p>}
    <ol className="space-y-3">{a.intervals.map((w, i) => <li key={i} className="rounded border p-3 text-sm">
      <p className="font-medium">{attentionLabel[w.attention]} · {utc(w.start)}{w.end !== w.start && ` — ${utc(w.end)}`}</p>
      <p>Peak density: {density(w.peak_density_kg_m3)} · peak ratio: {w.peak_ratio?.toFixed(2) ?? '—'} · {orbit ? 'forecast steps' : 'trajectory samples'}: {w.sample_count}</p>
      {w.sample_count > 0 && <p>{orbit ? 'Warning forecast steps span' : 'Warning samples span'} {w.sampled_duration_minutes ?? 0} minutes; {orbit ? 'spacecraft exposure is not established.' : 'continuous exposure between samples is not established.'}</p>}
      {w.peak_location && <p>{orbit ? 'Envelope maximum at' : 'Peak ratio at'} {utc(w.peak_location.time)} · {w.peak_location.altitude_km.toFixed(1)} km · latitude {w.peak_location.latitude_deg.toFixed(2)}°{w.peak_location.longitude_deg != null && ` · longitude ${w.peak_location.longitude_deg.toFixed(2)}°`} (density remains longitude-averaged).</p>}
      <ol className="mt-2 list-inside list-[upper-alpha] space-y-1">{w.reasons.map((r, j) => <li key={j}>{r.message}{r.source_id && <span className="block break-all text-xs text-muted-foreground">Source: {r.source_id}</span>}</li>)}</ol>
    </li>)}</ol>
    <details><summary className="cursor-pointer">{orbit ? 'Orbital envelope forecast values' : 'Trajectory sample values and coverage'}</summary>
      <div className="overflow-auto"><table className="w-full text-left text-sm"><thead><tr><th>UTC</th><th>Altitude / latitude</th><th>Forecast density</th><th>Reference</th><th>Ratio</th><th>Attention / reason</th></tr></thead>
        <tbody>{a.samples.slice(samplePage * 20, (samplePage + 1) * 20).map((p, i) => <tr key={i}><td className="whitespace-nowrap p-2">{utc(p.time)}</td><td>{p.altitude_km} km / {p.latitude_deg}°</td><td>{density(p.density_kg_m3)}</td><td>{density(p.reference_density_kg_m3)}</td><td>{p.ratio?.toFixed(2) ?? '—'}</td><td>{attentionLabel[p.attention]}{p.reasons.map((r, j) => <p key={j}>{r.message}</p>)}</td></tr>)}</tbody></table></div>
      <div className="mt-2 flex items-center gap-2"><Button variant="outline" disabled={!samplePage} onClick={() => setSamplePage(p => p - 1)}>Previous samples</Button><span>Page {samplePage + 1}</span><Button variant="outline" disabled={(samplePage + 1) * 20 >= a.samples.length} onClick={() => setSamplePage(p => p + 1)}>Next samples</Button></div>
    </details>
    <details><summary className="cursor-pointer">Assessment limitations and provenance</summary><ul className="list-inside list-disc space-y-1 text-sm">{a.limitations.map((s, i) => <li key={i}>{s}</li>)}</ul>
      <p className="mt-2 break-all text-xs">Run: {run.id} · density release: {a.source.density_release_id ?? '—'} · CME release: {a.source.cme_release_id ?? '—'}</p>
    </details>
  </Card>;
}

export default function DensityAdvisoryView() {
  const [items, setItems] = useState<DensityItem[]>([]);
  const [events, setEvents] = useState<DensityEventPage>({ items: [], total: 0 });
  const [page, setPage] = useState(1);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [version, setVersion] = useState(0);
  const [editor, setEditor] = useState<DensityProfile | null | undefined>(undefined);
  const [runs, setRuns] = useState<DensityRunSummary[]>([]);
  const [selected, setSelected] = useState<DensityRun | null>(null);
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<DensityItem['profile'] | null>(null);
  const activeProfile = items.find(item => item.profile.id === activeProfileId);
  const detailsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const [profiles, journal] = await Promise.all([
          apiRequest<DensityItem[]>(endpoint, { signal: controller.signal, cache: 'no-store' }),
          apiRequest<DensityEventPage>(`${endpoint}/events?page=${page}`, { signal: controller.signal, cache: 'no-store' }),
        ]);
        if (!controller.signal.aborted) { setItems(profiles); setEvents(journal); setError(''); setLoaded(true); }
      } catch (e) { if (!controller.signal.aborted) { setError(problem(e)); setItems([]); setEvents({ items: [], total: 0 }); setLoaded(true); } }
    }
    void load(); const timer = setInterval(() => void load(), 60000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [version, page]);
  useEffect(() => { if (selected) detailsRef.current?.focus(); }, [selected]);
  async function action(operation: () => Promise<unknown>) {
    setBusy(true); setError('');
    try { await operation(); setVersion(v => v + 1); }
    catch (e) { setError(problem(e)); }
    finally { setBusy(false); }
  }
  async function open(profileId: string, runId?: string) {
    const history = await apiRequest<DensityRunSummary[]>(`${endpoint}/${profileId}/history`);
    const selectedId = runId ?? history[0]?.id;
    const run = selectedId ? await apiRequest<DensityRun>(`${endpoint}/${profileId}/runs/${selectedId}`) : null;
    setRuns(run && !history.some(r => r.id === run.id) ? [{ id: run.id, profile_revision: run.profile_revision, computed_at: run.assessment.computed_at }, ...history] : history);
    setSelected(run ?? null);
  }
  return <div className="space-y-5">
    <PageHeading title="Drag risk" description="Density warnings across the altitude and latitude range of your orbit, relative to the published model background, with separate CME arrival context." action={<Button onClick={() => setEditor(null)}>Add spacecraft</Button>} />
    {error && <Message>{error}</Message>}
    {editor !== undefined && <Editor key={editor?.id ?? 'new'} profile={editor} busy={busy} cancel={() => setEditor(undefined)} save={async body => {
      setBusy(true);
      try { await apiRequest(editor ? `${endpoint}/${editor.id}?revision=${editor.revision}` : endpoint,
        { method: editor ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        setEditor(undefined); setVersion(v => v + 1);
      } finally { setBusy(false); }
    }} />}
    <Card className="overflow-auto p-4">{!loaded ? <p>Loading density advisories…</p> : !items.length ? <EmptyState title="No spacecraft" description="Add a spacecraft with its orbital parameters to begin." /> :
      <table className="w-full text-left text-sm" aria-label="Spacecraft status"><thead><tr><th className="p-3">Spacecraft</th><th className="p-3">Status</th><th className="p-3">Details</th><th className="p-3">Actions</th></tr></thead><tbody>{items.map(item => {
        const status = spacecraftStatus(item);
        return <tr key={item.profile.id} className="border-t align-top">
          <td className="p-3"><button className="font-medium text-primary hover:underline" disabled={busy} onClick={() => setActiveProfileId(item.profile.id)} title="Manage spacecraft">{item.profile.name}</button>
            <p className="mt-1 whitespace-nowrap text-xs text-muted-foreground" title="Periapsis / Apoapsis / Inclination">{item.profile.periapsis_km ?? '—'}km/{item.profile.apoapsis_km ?? '—'}km/{item.profile.inclination_deg ?? '—'}°</p>
          </td>
          <td className="p-3"><span className={`inline-flex rounded px-2 py-1 text-xs font-medium ${status.warning ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400' : 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'}`}>{status.warning ? 'Warning' : 'OK'}</span></td>
          <td className="p-3">{status.details.length ? status.details.map(detail => <p key={detail}>{detail}</p>) : 'No warnings.'}</td>
          <td className="p-3"><div className="flex gap-2">
            <Button size="sm" variant="outline" disabled={busy} onClick={() => void action(async () => setEditor(await apiRequest<DensityProfile>(`${endpoint}/${item.profile.id}`)))}>Edit</Button>
            <Button size="sm" variant="outline" className="text-destructive" disabled={busy} onClick={() => { setError(''); setDeleting(item.profile); }}>Delete</Button>
          </div></td>
        </tr>;
      })}</tbody></table>}
    </Card>
    <Dialog open={deleting !== null} onOpenChange={open => { if (!open && !busy) setDeleting(null); }}>
      <DialogContent>
        <DialogHeader><DialogTitle>Delete {deleting?.name}?</DialogTitle><DialogDescription>The spacecraft, its saved forecasts and journal entries will be deleted.</DialogDescription></DialogHeader>
        {error && <Message>{error}</Message>}
        <DialogFooter>
          <Button variant="outline" disabled={busy} onClick={() => setDeleting(null)}>Cancel</Button>
          <Button variant="destructive" disabled={busy || !deleting} onClick={() => {
            if (!deleting) return;
            const target = deleting;
            void action(async () => {
              await apiRequest(`${endpoint}/${target.id}?revision=${target.revision}`, { method: 'DELETE' });
              setItems(previous => previous.filter(item => item.profile.id !== target.id));
              if (editor?.id === target.id) setEditor(undefined);
              if (selected?.profile_id === target.id) { setSelected(null); setRuns([]); }
              if (activeProfileId === target.id) setActiveProfileId(null);
              setPage(1); setDeleting(null);
            });
          }}>Delete spacecraft</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    {activeProfile && <Card className="space-y-3 p-5">
      <div className="flex items-center justify-between gap-3"><h2 className="font-semibold">{activeProfile.profile.name}</h2><Button variant="ghost" onClick={() => setActiveProfileId(null)}>Close</Button></div>
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" disabled={busy} onClick={() => void action(async () => { await apiRequest(`${endpoint}/${activeProfile.profile.id}/assess`, { method: 'POST' }); await open(activeProfile.profile.id); })}>Assess</Button>
        <Button variant="outline" disabled={busy || !activeProfile.latest} onClick={() => void action(() => open(activeProfile.profile.id))}>View forecast</Button>
        <Button variant="outline" disabled={busy} onClick={() => void action(async () => setEditor(await apiRequest<DensityProfile>(`${endpoint}/${activeProfile.profile.id}`)))}>Update spacecraft</Button>
      </div>
    </Card>}
    {selected && <div ref={detailsRef} tabIndex={-1} className="min-w-0 space-y-3 outline-none">
      <label>Saved forecast history<select className={inputClass} value={selected.id} disabled={busy} onChange={e => void action(async () => setSelected(await apiRequest<DensityRun>(`${endpoint}/${selected.profile_id}/runs/${e.target.value}`)))}>{runs.map(r => <option key={r.id} value={r.id}>{utc(r.computed_at)} · v{r.profile_revision}</option>)}</select></label>
      <RunDetails key={selected.id} run={selected} />
    </div>}
    <Card className="space-y-3 p-5"><h2 className="text-lg font-semibold">Density advisory journal</h2><p className="text-sm text-muted-foreground">Attention and data-status changes; each entry opens its immutable source forecast.</p>
      {!events.items.length && <p>No recorded changes.</p>}
      <ul className="space-y-3">{events.items.map(e => <li key={e.id} className="rounded border p-3 text-sm"><p>{utc(e.detected_at)} · {e.message}</p><div className="mt-2 flex gap-2">
        <Button variant="outline" disabled={busy} onClick={() => void action(() => open(e.profile_id, e.run_id))}>Open source forecast</Button>
        {e.acknowledged_at ? <span>Reviewed {utc(e.acknowledged_at)}</span> : <Button variant="outline" disabled={busy} onClick={() => void action(() => apiRequest(`${endpoint}/events/${e.id}/ack`, { method: 'POST' }))}>Mark reviewed</Button>}
      </div></li>)}</ul>
      <div className="flex items-center gap-2"><Button variant="outline" disabled={page === 1} onClick={() => setPage(p => p - 1)}>Previous events</Button><span>Page {page}</span><Button variant="outline" disabled={page * 20 >= events.total} onClick={() => setPage(p => p + 1)}>Next events</Button></div>
    </Card>
  </div>;
}
