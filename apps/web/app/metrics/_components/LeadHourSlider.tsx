'use client';
import { useState, useTransition } from 'react';
import local from './metrics.module.css';

export default function LeadHourSlider({ id, hours, hour, onChange }: {
  id: string; hours: number[]; hour: number; onChange: (hour: number) => void;
}) {
  const [preview, setPreview] = useState(hour);
  const [syncedHour, setSyncedHour] = useState(hour);
  const [pending, startTransition] = useTransition();
  // Reconcile external chart/horizon changes without an extra effect render.
  if (!pending && syncedHour !== hour) {
    setSyncedHour(hour);
    setPreview(hour);
  }
  const current = hours.includes(preview) ? preview : hour;
  return <>
    <label htmlFor={id}>Evaluation lead hour <output htmlFor={id}>+{current}h</output></label>
    <input className={local.leadSlider} id={id} type="range" min={0} max={hours.length - 1} step={1}
      value={hours.indexOf(current)} aria-valuetext={`+${current} hours`} disabled={hours.length < 2}
      onChange={event => {
        const next = hours[Number(event.target.value)];
        setPreview(next);
        startTransition(() => onChange(next));
      }} />
    <div className={local.leadScale} aria-hidden="true"><span>+{hours[0]}h</span><span>+{hours.at(-1)}h</span></div>
  </>;
}
