'use client';
import { useState } from 'react';
import Link from 'next/link';
import { ArrowUpRight, Layers3 } from 'lucide-react';
import WorkspaceShell from './WorkspaceShell';
import ForecastBoard, { forecastNumber, type ForecastResource } from './ForecastBoard';
import { useResource } from '../_utils/useResource';
import type { Forecast } from '../_utils/api';
import { productsBySlug } from '../_config/products';
import { formatProbability } from '../_utils/forecast';
import styles from './forecast.module.css';

const targets = [
  { slug: 'solar-wind-speed', key: 'v', label: 'Solar wind speed', unit: 'km/s', color: '#79b9e6' },
  { slug: 'solar-wind-density', key: 'n', label: 'Proton density', unit: 'cm⁻³', color: '#b1a0df' },
  { slug: 'geomagnetic-activity', key: 'kp', label: 'Geomagnetic activity', unit: '', color: '#d2b984' },
  { slug: 'dst', key: 'dst', label: 'Dst index', unit: 'nT', color: '#73b9bc' },
];
export default function ForecastOverview() {
  const speed = useResource<Forecast>('/public/forecasts/solar-wind-speed');
  const density = useResource<Forecast>('/public/forecasts/solar-wind-density');
  const geomagnetic = useResource<Forecast>('/public/forecasts/geomagnetic-activity');
  const dst = useResource<Forecast>('/public/forecasts/dst');
  const resources: Record<string, ForecastResource> = { 'solar-wind-speed': speed, 'solar-wind-density': density, 'geomagnetic-activity': geomagnetic, dst };
  const [selected, setSelected] = useState('solar-wind-speed');
  return <WorkspaceShell section="Forecast overview">
    <main id="forecast-content" className={styles.page}>
      <div className={styles.pageHeading}><div><div className={styles.eyebrow}>EARTH–SUN ENVIRONMENT <span>/ 02</span></div><h1>Forecast overview</h1><p>Published outlooks for solar wind and geomagnetic activity.</p></div>
        <Link className={styles.headingLink} href="/products"><Layers3 size={14} aria-hidden="true" />All forecast products<ArrowUpRight size={13} aria-hidden="true" /></Link>
      </div>
      <section className={styles.summary} aria-label="Forecast products summary">
        <div className={styles.summaryGrid}>{targets.map(target => {
          const { data, error } = resources[target.slug];
          const first = data?.predictions[0];
          const values = first?.variables[target.key];
          const probability = values?.binary.find(item => item.threshold === 5 && (!item.operator || item.operator === 'gte'))?.probability;
          const value = target.key === 'kp' ? probability : values?.continuous?.q50;
          return <button key={target.slug} className={styles.summaryCard} aria-label={`Show ${target.label} forecast`} aria-pressed={selected === target.slug} onClick={() => setSelected(target.slug)} style={{ '--forecast-accent': target.color } as React.CSSProperties}>
            <span className={styles.summaryLabel}>{target.label}<span>{target.key.toUpperCase()}</span></span>
            <span className={styles.summaryValue}>{target.key === 'kp' ? formatProbability(probability) : forecastNumber(value)} <small>{target.unit}</small></span>
            <span className={styles.summaryMeaning}>{target.key === 'kp' ? 'Kp ≥ 5 probability' : 'Median · q50'}{first ? ` · lead +${first.lead_hours}h` : ''}</span>
            <span className={styles.summaryState} data-warning={!!error}><i />{data ? value == null ? 'No value at first time' : 'Published' : error ? 'Unavailable' : 'Loading…'}</span>
            <span className={styles.summaryTime}>{data ? `Issued ${data.issue_time.slice(5, 16).replace('T', ' ')} UTC` : 'Awaiting release data'}</span>
          </button>;
        })}</div>
        <div className={styles.summaryCaption}><span>First forecast time in each release</span><span>Products retain their own issue times · select a product to explore</span></div>
      </section>
      <ForecastBoard key={selected} product={productsBySlug[selected]} resource={resources[selected]} overview />
      <div className={styles.pageFooter}><span>Model forecasts · all times UTC</span><Link href="/live">Compare with live observations<ArrowUpRight size={12} aria-hidden="true" /></Link></div>
    </main>
  </WorkspaceShell>;
}
