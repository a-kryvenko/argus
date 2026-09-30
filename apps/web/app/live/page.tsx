'use client';

import { useMemo, useState } from 'react';
import { ArrowUpRight, Clock3 } from 'lucide-react';
import WorkspaceShell from '../_components/WorkspaceShell';
import GeomagneticLive from './GeomagneticLive';
import ObservationSummary from './ObservationSummary';
import ObservationInspector from './ObservationInspector';
import CollectionStatus from './CollectionStatus';
import SolarWindLive from './SolarWindLive';
import HourlyObservations from './HourlyObservations';
import type { History } from './solarWind';
import type { IndexHistory } from './geomagnetic';
import { stamp, type Selection } from './observations';
import styles from './page.module.css';
import { useLiveQuery, useLiveClock, type Summary } from './useLiveQuery';

export default function Live() {
  const [hours, setHours] = useState(24);
  const [selected, setSelected] = useState<Selection>('bz');
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const now = useLiveClock();
  const snapshot = { ...useLiveQuery<Summary>('/public/observations/summary'), now };
  const minute = now === undefined ? undefined : Math.floor(now / 60000);
  const window = useMemo(() => minute === undefined ? undefined : {
    from: new Date(minute * 60000 - hours * 3600000).toISOString(),
    to: new Date(minute * 60000).toISOString(),
  }, [minute, hours]);
  const windHistory = useLiveQuery<History>(window ? '/public/observations/solar-wind/history?metrics=bx,by,bz,bt,v,n,t&resolution=auto' : null, window);
  const indexHistory = useLiveQuery<IndexHistory>(window ? '/public/observations/geomagnetic/history' : null, window);
  const select = (metric: Selection) => { setSelected(metric); setInspectorOpen(true); };
  const snapshotOld = now && snapshot.receivedAt ? now - snapshot.receivedAt > 120000 : false;
  return <WorkspaceShell section="Observations" contentId="live-content"
    status={snapshot.failed ? 'Refresh delayed' : snapshotOld ? 'Old snapshot' : 'Auto refresh · 60 s'}
    statusTone={snapshot.failed || snapshotOld ? 'warning' : 'good'}>
      <main id="live-content" className={styles.page}>
        <div className={styles.pageHeading}>
          <div><div className={styles.eyebrow}>EARTH–SUN ENVIRONMENT <span>/ 01</span></div><h1>Live observations</h1><p>Solar wind upstream. Geomagnetic response on Earth.</p></div>
          <div className={styles.snapshotTime}><span>LATEST SNAPSHOT · UTC</span><time dateTime={snapshot.data?.generated_at}>{stamp(snapshot.data?.generated_at)}</time></div>
        </div>
        <ObservationSummary snapshot={snapshot} selected={selected} onSelect={select} />
        <div className={styles.toolbar}>
          <div className={styles.windowLabel}><Clock3 size={14} aria-hidden="true" /><span>HISTORY WINDOW</span><span className={styles.windowRange}>{window ? `${stamp(window.from)} → ${stamp(window.to)} UTC` : 'Preparing time window…'}</span></div>
          <div className={styles.periods} role="group" aria-label="History period">
            {[6, 24, 72, 168, 720].map(period => <button key={period} aria-pressed={hours === period} aria-label={period < 48 ? `${period} hours` : `${period / 24} days`} onClick={() => setHours(period)}>{period < 48 ? `${period}h` : `${period / 24}d`}</button>)}
          </div>
        </div>
        <div className={styles.workspaceGrid}>
          <div className={styles.historyWorkspace}>
            <SolarWindLive hours={hours} history={windHistory} selected={selected} onSelect={select} />
            <GeomagneticLive history={indexHistory} selected={selected} onSelect={select} />
            <div className={styles.supportPanels}>
              <CollectionStatus now={now} />
              <details className={styles.additional}>
                <summary>Additional hourly indices <span>Ap · F10.7 · S10 · M10 · Y10</span></summary>
                <HourlyObservations />
              </details>
            </div>
            <div className={styles.workspaceFooter}><span>All times UTC · operational observations</span><a href="/api/v1/public/observations/summary">Observation summary <ArrowUpRight size={12} aria-hidden="true" /></a></div>
          </div>
          <ObservationInspector selected={selected} onSelect={select} snapshot={snapshot} windHistory={windHistory} indexHistory={indexHistory} expanded={inspectorOpen} onToggle={() => setInspectorOpen(open => !open)} />
        </div>
      </main>
  </WorkspaceShell>;
}
