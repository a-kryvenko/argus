import SolarWindLive from './SolarWindLive';
import HourlyObservations from './HourlyObservations';
import styles from './page.module.css';

export default function Live() {
  return <main className={`container ${styles.page}`}>
    <h1>Live observations</h1>
    <SolarWindLive />
    <details className={styles.additional}>
      <summary>Additional hourly indices</summary>
      <HourlyObservations />
    </details>
  </main>;
}
