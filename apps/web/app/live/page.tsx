'use client';

import { useState } from 'react';
import GeomagneticLive from './GeomagneticLive';
import ObservationSummary from './ObservationSummary';
import CollectionStatus from './CollectionStatus';
import SolarWindLive from './SolarWindLive';
import HourlyObservations from './HourlyObservations';
import styles from './page.module.css';
import { useLiveQuery, useLiveClock, type Summary } from './useLiveQuery';

export default function Live() {
  const [hours, setHours] = useState(24);
  const now = useLiveClock();
  const snapshot = { ...useLiveQuery<Summary>('/public/observations/summary'), now };
  return <main className={`container ${styles.page}`}>
    <h1>Live observations</h1>
    <CollectionStatus now={now} />
    <ObservationSummary snapshot={snapshot} />
    <SolarWindLive hours={hours} onHoursChange={setHours} snapshot={snapshot} />
    <GeomagneticLive hours={hours} onHoursChange={setHours} snapshot={snapshot} />
    <details className={styles.additional}>
      <summary>Additional hourly indices</summary>
      <HourlyObservations />
    </details>
  </main>;
}
