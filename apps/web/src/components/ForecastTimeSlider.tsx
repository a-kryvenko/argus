'use client';
import { useState, useTransition } from 'react';
import type { ForecastPoint } from '@/forecast/types';
import { formatForecastTime } from '../forecast/format';
import styles from './range.module.css';

export default function ForecastTimeSlider({ id, predictions, time, onChange }: {
  id: string; predictions: Pick<ForecastPoint, 'valid_time' | 'lead_hours'>[];
  time?: string; onChange: (time: string) => void;
}) {
  const [preview, setPreview] = useState(time);
  const [syncedTime, setSyncedTime] = useState(time);
  const [pending, startTransition] = useTransition();
  if (!pending && syncedTime !== time) {
    setSyncedTime(time);
    setPreview(time);
  }
  const current = predictions.find(point => point.valid_time === preview)
    ?? predictions.find(point => point.valid_time === time) ?? predictions[0];
  return <>
    <label htmlFor={id}>Forecast time · UTC<output className={styles.timestamp} htmlFor={id}>{current ? formatForecastTime(current.valid_time) : 'No forecast times'}</output></label>
    <input id={id} className={styles.slider} type="range" min={0} max={Math.max(0, predictions.length - 1)} step={1}
      value={current ? predictions.indexOf(current) : 0} disabled={predictions.length < 2}
      aria-label="Forecast time · UTC" aria-valuetext={current ? `${formatForecastTime(current.valid_time)} · +${current.lead_hours} hours` : 'No forecast times'}
      onChange={event => {
        const point = predictions[Number(event.target.value)];
        if (!point) return;
        setPreview(point.valid_time);
        startTransition(() => onChange(point.valid_time));
      }} />
    {!!predictions.length && <div className={styles.scale} aria-hidden="true"><span>+{predictions[0].lead_hours}h</span><span>+{predictions.at(-1)!.lead_hours}h</span></div>}
  </>;
}
