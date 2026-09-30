'use client';
import type { ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, BookOpen, ChartNoAxesCombined, ChevronRight, CircleHelp, Clock3, Gauge, LayoutGrid, Menu, Orbit, Radio, ShieldCheck } from 'lucide-react';
import { useClock } from '../_utils/useClock';
import styles from './workspace.module.css';

const navigation = [
  { href: '/', label: 'Forecast overview', icon: LayoutGrid },
  { href: '/live', label: 'Live observations', icon: Radio },
  { href: '/products', label: 'Forecast products', icon: Activity },
  { href: '/metrics', label: 'Model performance', icon: ChartNoAxesCombined },
  { href: '/dashboard/risk/leo', label: 'LEO drag', icon: Orbit },
];

export default function WorkspaceShell({ children, section, contentId = 'forecast-content', status = 'Published model outputs', statusTone = 'neutral' }: {
  children: ReactNode; section: string; contentId?: string; status?: string; statusTone?: 'neutral' | 'good' | 'warning';
}) {
  const pathname = usePathname();
  const now = useClock();
  return <div className={styles.workspace}>
    <a className={styles.skipLink} href={`#${contentId}`}>Skip to {section.toLowerCase()}</a>
    <aside className={styles.sidebar} aria-label="Workspace navigation">
      <Link className={styles.brand} href="/" aria-label="Argus SunWatch home"><Orbit size={28} strokeWidth={1.3} aria-hidden="true" /><span>ARGUS<small>SUNWATCH</small></span></Link>
      <div className={styles.navLabel}>WORKSPACE</div>
      <nav className={styles.navigation} aria-label="Main navigation">
        {navigation.map(({ href, label, icon: Icon }) => {
          const active = href === '/' ? pathname === '/' : pathname === href || pathname.startsWith(`${href}/`);
          return <Link href={href} key={href} aria-label={label} title={label} aria-current={active ? 'page' : undefined}>
            <Icon size={17} strokeWidth={1.5} aria-hidden="true" /><span>{label}</span>{active && <span className={styles.navMarker} />}
          </Link>;
        })}
      </nav>
      <div className={styles.sidebarBottom}>
        <div className={styles.navLabel}>RESOURCES</div>
        <nav className={styles.navigation} aria-label="Resources">
          <Link href="/dashboard" aria-label="Dashboard" title="Dashboard"><Gauge size={17} strokeWidth={1.5} aria-hidden="true" /><span>Dashboard</span></Link>
          <Link href="/help" aria-label="Help" title="Help"><CircleHelp size={17} strokeWidth={1.5} aria-hidden="true" /><span>Help & methodology</span></Link>
          <a href="/api/v1/docs" aria-label="API documentation" title="API documentation"><BookOpen size={17} strokeWidth={1.5} aria-hidden="true" /><span>API documentation</span></a>
        </nav>
        <div className={styles.sidebarFoot}><ShieldCheck size={14} aria-hidden="true" /><span>SPACE WEATHER<br />ANALYSIS WORKSPACE</span></div>
      </div>
    </aside>
    <div className={styles.workspaceMain}>
      <header className={styles.commandBar}>
        <div className={styles.breadcrumb}><span>ARGUS</span><ChevronRight size={12} aria-hidden="true" /><span>Space weather</span><ChevronRight size={12} aria-hidden="true" /><strong>{section}</strong></div>
        <details className={styles.mobileResources}>
          <summary><Menu size={13} aria-hidden="true" />Resources</summary>
          <nav aria-label="Mobile resources"><Link href="/dashboard">Dashboard</Link><Link href="/help">Help & methodology</Link><a href="/api/v1/docs">API documentation</a></nav>
        </details>
        <div className={styles.commandStatus}><span className={styles.refreshState} data-tone={statusTone} data-warning={statusTone === 'warning'}><span className={styles.dot} />{status}</span><span className={styles.utcClock}><Clock3 size={12} aria-hidden="true" />{now ? new Date(now).toISOString().slice(11, 16) : '—'} UTC</span></div>
      </header>
      {children}
    </div>
  </div>;
}
