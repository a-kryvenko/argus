'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import { Activity, ArrowUpRight, BookOpen, ChartNoAxesCombined, ChevronRight, CircleHelp, Clock3, Gauge, LayoutGrid, Menu, Orbit, Radio, ShieldCheck } from 'lucide-react';
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

const navigation = [
  { href: '/', label: 'Forecast overview', icon: LayoutGrid },
  { href: '/live', label: 'Live observations', icon: Radio },
  { href: '/products', label: 'Forecast products', icon: Activity },
  { href: '/metrics', label: 'Model performance', icon: ChartNoAxesCombined },
  { href: '/dashboard/risk/leo', label: 'LEO drag', icon: Orbit },
];

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
  return <div className={styles.workspace}>
    <a className={styles.skipLink} href="#live-content">Skip to observations</a>
    <aside className={styles.sidebar} aria-label="Workspace navigation">
      <Link className={styles.brand} href="/" aria-label="Argus SunWatch home"><Orbit size={28} strokeWidth={1.3} aria-hidden="true" /><span>ARGUS<small>SUNWATCH</small></span></Link>
      <div className={styles.navLabel}>WORKSPACE</div>
      <nav className={styles.navigation} aria-label="Main navigation">
        {navigation.map(({ href, label, icon: Icon }) => <Link href={href} key={href} aria-label={label} title={label} aria-current={href === '/live' ? 'page' : undefined}>
          <Icon size={17} strokeWidth={1.5} aria-hidden="true" /><span>{label}</span>{href === '/live' && <span className={styles.navMarker} />}
        </Link>)}
      </nav>
      <div className={styles.sidebarBottom}>
        <div className={styles.navLabel}>RESOURCES</div>
        <nav className={styles.navigation} aria-label="Resources">
          <Link href="/dashboard" aria-label="Dashboard" title="Dashboard"><Gauge size={17} strokeWidth={1.5} aria-hidden="true" /><span>Dashboard</span></Link>
          <Link href="/help" aria-label="Help" title="Help"><CircleHelp size={17} strokeWidth={1.5} aria-hidden="true" /><span>Help & methodology</span></Link>
          <a href="/api/v1/docs" aria-label="API documentation" title="API documentation"><BookOpen size={17} strokeWidth={1.5} aria-hidden="true" /><span>API documentation</span></a>
        </nav>
        <div className={styles.sidebarFoot}><ShieldCheck size={14} aria-hidden="true" /><span>SPACE WEATHER<br />OBSERVATION WORKSPACE</span></div>
      </div>
    </aside>
    <div className={styles.workspaceMain}>
      <header className={styles.commandBar}>
        <div className={styles.breadcrumb}><span>ARGUS</span><ChevronRight size={12} aria-hidden="true" /><span>Space weather</span><ChevronRight size={12} aria-hidden="true" /><strong>Observations</strong></div>
        <details className={styles.mobileResources}>
          <summary><Menu size={13} aria-hidden="true" />Resources</summary>
          <nav aria-label="Mobile resources"><Link href="/dashboard">Dashboard</Link><Link href="/help">Help & methodology</Link><a href="/api/v1/docs">API documentation</a></nav>
        </details>
        <div className={styles.commandStatus}><span className={styles.refreshState} data-warning={snapshot.failed || snapshotOld}><span className={styles.dot} />{snapshot.failed ? 'Refresh delayed' : snapshotOld ? 'Old snapshot' : 'Auto refresh · 60 s'}</span><span className={styles.utcClock}><Clock3 size={12} aria-hidden="true" />{now ? new Date(now).toISOString().slice(11, 16) : '—'} UTC</span></div>
      </header>
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
    </div>
  </div>;
}
