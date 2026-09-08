'use client';

import { useState } from 'react';
import GeomagneticLive from './GeomagneticLive';
import ObservationSummary from './ObservationSummary';
import CollectionStatus from './CollectionStatus';
import SolarWindLive from './SolarWindLive';
import HourlyObservations from './HourlyObservations';
import styles from './page.module.css';

export default function Live() {
  const [hours, setHours] = useState(24);
  return <main className={`container ${styles.page}`}>
    <h1>Live observations</h1>
    <CollectionStatus />
    <ObservationSummary />
    <SolarWindLive hours={hours} onHoursChange={setHours} />
    <GeomagneticLive hours={hours} onHoursChange={setHours} />
    <details className={styles.additional}>
      <summary>Additional hourly indices</summary>
      <HourlyObservations />
    </details>
  </main>;
}
