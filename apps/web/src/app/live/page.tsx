'use client';
import { useClock } from '@/hooks/useClock';

import { useMemo, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import WorkspaceShell from '../../components/WorkspaceShell';
import GeomagneticLive from './GeomagneticLive';
import ObservationSummary from './ObservationSummary';
import SolarWindLive from './SolarWindLive';
import HourlyObservations from './HourlyObservations';
import type { History } from './solarWind';
import type { IndexHistory } from './geomagnetic';
import styles from './page.module.css';
import { useLiveQuery, type Summary } from './useLiveQuery';

export default function Live() {
  const [hours, setHours] = useState(24);
  const [hourlyOpen, setHourlyOpen] = useState(false);
  const now = useClock();
  const snapshot = { ...useLiveQuery<Summary>('/observations?trends=true'), now };
  const minute = now === undefined ? undefined : Math.floor(now / 60000);
  const window = useMemo(() => minute === undefined ? undefined : {
    from: new Date(minute * 60000 - hours * 3600000).toISOString(),
    to: new Date(minute * 60000).toISOString(),
  }, [minute, hours]);
  const history = useLiveQuery<History & IndexHistory>(window ? '/observations/history?observations=bx,by,bz,bt,v,n,t,kp,dst&resolution=auto' : null, window);
  const snapshotOld = now && snapshot.receivedAt ? now - snapshot.receivedAt > 120000 : false;
  return <WorkspaceShell section="Observations" contentId="live-content"
    status={snapshot.failed ? 'Refresh delayed' : snapshotOld ? 'Old snapshot' : 'Auto refresh · 60 s'}
    statusTone={snapshot.failed || snapshotOld ? 'warning' : 'good'}>
      <main id="live-content" className={styles.page}>
        <div className={styles.pageHeading}>
          <div><h1>Live space weather</h1></div>
          <div className={styles.liveStatus}><span data-warning={snapshot.failed || !!snapshotOld}><span className={styles.dot} />{snapshot.failed ? 'Refresh delayed' : snapshotOld ? 'Old snapshot' : 'Live · updates every 60 s'}</span><a className={styles.snapshotApi} href="/api/v1/observations?trends=true" aria-label="Observations API (JSON)">API <ArrowUpRight size={11} aria-hidden="true" /></a></div>
        </div>
        <ObservationSummary snapshot={snapshot} />
        <div className={styles.toolbar}>
          <div className={styles.windowLabel}><h2 className={styles.windowTitle}>History</h2></div>
          <div className={styles.toolbarActions}><div className={styles.periods} role="group" aria-label="History period">
            {[6, 24, 72, 168, 720].map(period => <button key={period} aria-pressed={hours === period} aria-label={period < 48 ? `${period} hours` : `${period / 24} days`} onClick={() => setHours(period)}>{period < 48 ? `${period}h` : `${period / 24}d`}</button>)}
          </div>
          </div>
        </div>
        <div className={styles.workspaceGrid}>
          <div className={styles.historyWorkspace}>
            <SolarWindLive hours={hours} history={history} />
            <GeomagneticLive history={history} />
            <div className={styles.supportPanels}>
              <details className={styles.additional} onToggle={event => setHourlyOpen(event.currentTarget.open)}>
                <summary>Additional hourly indices <span>Ap · F10.7 · S10 · M10 · Y10</span></summary>
                {hourlyOpen && <HourlyObservations />}
              </details>
            </div>
            <div className={styles.workspaceFooter}><span>All times UTC · operational observations</span><a href="/api/v1/observations">Observation summary <ArrowUpRight size={12} aria-hidden="true" /></a></div>
          </div>
        </div>
      </main>
  </WorkspaceShell>;
}
