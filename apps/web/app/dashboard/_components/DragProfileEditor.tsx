'use client';
import { useState, type FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Message, selectClass } from './presentation';
import type { Profile, ProfileInput } from './drag-monitor';

export default function DragProfileEditor({ profile, busy, onSave, onCancel }: {
  profile: Profile | null; busy: boolean;
  onSave: (body: ProfileInput) => Promise<void>; onCancel: () => void;
}) {
  const [count, setCount] = useState(profile?.modes.length ?? 1);
  const [error, setError] = useState('');
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const value = (key: string) => String(form.get(key) ?? '');
    const numeric = (key: string) => Number(value(key));
    const modes = Array.from({ length: count }, (_, i) => ({ name: value(`mode-${i}`).trim(),
      effective_area_m2: numeric(`area-${i}`), drag_coefficient: numeric(`cd-${i}`) }));
    if (new Set(modes.map(mode => mode.name)).size !== count) {
      setError('Each attitude mode needs a unique name.'); return;
    }
    if (numeric('high') <= numeric('elevated')) {
      setError('The high limit must exceed the review limit.'); return;
    }
    if (Boolean(value('growth_percent')) !== Boolean(value('growth_min_m'))) {
      setError('Enter both growth thresholds, or leave both empty for service defaults.'); return;
    }
    setError('');
    await onSave({ name: value('name').trim(), altitude_km: numeric('altitude_km'),
      inclination_deg: numeric('inclination_deg'), mass_kg: numeric('mass_kg'),
      horizon_hours: numeric('horizon_hours') as 24 | 48,
      thresholds: { elevated_altitude_loss_m: numeric('elevated'), high_altitude_loss_m: numeric('high') },
      modes, active_mode: modes[numeric('active_mode')].name, enabled: profile?.enabled ?? true,
      event_policy: value('growth_percent') ? { growth_percent: numeric('growth_percent'), growth_min_m: numeric('growth_min_m') } : null });
  }
  return <Card className="p-5">
    <h2 className="mb-3 font-medium">{profile ? `Edit ${profile.name}` : 'Add spacecraft'}</h2>
    <p className="mb-4 text-sm text-muted-foreground">Enter your spacecraft parameters and operational limits. Circular orbits only; no maneuvers are modeled.</p>
    {error && <Message>{error}</Message>}
    <form onSubmit={submit}><fieldset disabled={busy} className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <div className="space-y-2"><Label htmlFor="profile-name">Spacecraft name</Label><Input id="profile-name" name="name" required maxLength={120} defaultValue={profile?.name} /></div>
        {([
          ['altitude_km', 'Mean altitude (km)', 200, 800],
          ['inclination_deg', 'Orbit inclination (°)', 0, 180],
          ['mass_kg', 'Spacecraft mass (kg)', 0.01, 1e7],
        ] as const).map(([name, label, min, max]) => <div className="space-y-2" key={name}>
          <Label htmlFor={`profile-${name}`}>{label}</Label><Input id={`profile-${name}`} name={name} type="number" required step="any" min={min} max={max} defaultValue={profile?.[name]} />
        </div>)}
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="space-y-2"><Label htmlFor="profile-horizon">Limit horizon</Label><select id="profile-horizon" name="horizon_hours" defaultValue={profile?.horizon_hours ?? 24} className={selectClass}><option value="24">24 hours</option><option value="48">48 hours</option></select></div>
        <div className="space-y-2"><Label htmlFor="profile-elevated">Review at total altitude loss (m)</Label><Input id="profile-elevated" name="elevated" type="number" required min="0.000001" step="any" defaultValue={profile?.thresholds.elevated_altitude_loss_m} /></div>
        <div className="space-y-2"><Label htmlFor="profile-high">High at total altitude loss (m)</Label><Input id="profile-high" name="high" type="number" required min="0.000001" step="any" defaultValue={profile?.thresholds.high_altitude_loss_m} /></div>
      </div>
      <p className="text-xs text-muted-foreground">Limits apply to cumulative loss over the selected horizon, starting at the atmosphere release time. They are user rules, not calibrated probabilities.</p>
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2"><Label htmlFor="profile-growth-percent">Growth event threshold (%)</Label><Input id="profile-growth-percent" name="growth_percent" type="number" min="0.000001" max="10000" step="any" placeholder="Service default" defaultValue={profile?.event_policy?.growth_percent} /></div>
        <div className="space-y-2"><Label htmlFor="profile-growth-min">Minimum growth for event (m)</Label><Input id="profile-growth-min" name="growth_min_m" type="number" min="0.000001" step="any" placeholder="Service default" defaultValue={profile?.event_policy?.growth_min_m} /></div>
      </div>
      <p className="text-xs text-muted-foreground">Both growth thresholds must be reached. Leave both empty to use service defaults.</p>
      {Array.from({ length: count }, (_, i) => <div key={i} className="grid gap-3 rounded-lg border p-3 sm:grid-cols-3">
        <div className="space-y-2"><Label htmlFor={`mode-${i}`}>Mode {i + 1} name</Label><Input id={`mode-${i}`} name={`mode-${i}`} required maxLength={80} defaultValue={profile?.modes[i]?.name ?? (i === 0 ? 'Nominal' : '')} /></div>
        <div className="space-y-2"><Label htmlFor={`area-${i}`}>Mode {i + 1} effective area (m²)</Label><Input id={`area-${i}`} name={`area-${i}`} type="number" required min="0.000001" max="1000000" step="any" defaultValue={profile?.modes[i]?.effective_area_m2} /></div>
        <div className="space-y-2"><Label htmlFor={`cd-${i}`}>Mode {i + 1} drag coefficient</Label><Input id={`cd-${i}`} name={`cd-${i}`} type="number" required min="0.000001" max="10" step="any" defaultValue={profile?.modes[i]?.drag_coefficient} /></div>
      </div>)}
      <div className="flex flex-wrap items-end gap-3">
        <Button type="button" variant="outline" disabled={count >= 5} onClick={() => setCount(count + 1)}>Add attitude mode</Button>
        {count > 1 && <Button type="button" variant="outline" onClick={() => setCount(count - 1)}>Remove last mode</Button>}
        <div className="space-y-2"><Label htmlFor="profile-active">Active attitude mode</Label><select id="profile-active" name="active_mode" className={selectClass} defaultValue={Math.max(0, profile?.modes.findIndex(mode => mode.name === profile.active_mode) ?? 0)}>
          {Array.from({ length: count }, (_, i) => <option key={i} value={i}>Mode {i + 1}</option>)}
        </select></div>
      </div>
      <div className="flex gap-3"><Button type="submit">{busy ? 'Saving…' : 'Save spacecraft'}</Button><Button type="button" variant="outline" onClick={onCancel}>Cancel</Button></div>
    </fieldset></form>
  </Card>;
}
