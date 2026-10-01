'use client';
import type { ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, BookOpen, ChartNoAxesCombined, ChevronRight, CircleHelp, Clock3, Gauge, LayoutGrid, Menu, LogOut, Orbit, Radio, ShieldCheck, type LucideIcon } from 'lucide-react';
import { useClock } from '../_utils/useClock';
import styles from './workspace.module.css';

const navigation = [
  { href: '/', label: 'Forecast overview', icon: LayoutGrid },
  { href: '/live', label: 'Live observations', icon: Radio },
  { href: '/products', label: 'Forecast products', icon: Activity },
  { href: '/metrics', label: 'Model performance', icon: ChartNoAxesCombined },
  { href: '/dashboard/risk/leo', label: 'LEO drag', icon: Orbit },
];

export type WorkspaceNavigationGroup = { title: string; items: { href: string; title: string; icon: LucideIcon }[] };
type WorkspaceAccount = { username: string; role: string; signingOut: boolean; logout: () => void };
function Account({ account }: { account: WorkspaceAccount }) {
  return <details className={styles.account}>
    <summary role="button" aria-label="Account menu" title={account.username}><ShieldCheck size={16} aria-hidden="true" /><span>{account.username}<small>{account.role}</small></span></summary>
    <button aria-label={account.signingOut ? 'Signing out…' : 'Sign out'} disabled={account.signingOut} onClick={account.logout}><LogOut size={14} aria-hidden="true" /><span>{account.signingOut ? 'Signing out…' : 'Sign out'}</span></button>
  </details>;
}

export default function WorkspaceShell({ children, section, contentId = 'forecast-content', status = 'Published model outputs', statusTone = 'neutral', navigationGroups = [], account }: {
  children: ReactNode; section: string; contentId?: string; status?: string; statusTone?: 'neutral' | 'good' | 'warning'; navigationGroups?: WorkspaceNavigationGroup[]; account?: WorkspaceAccount;
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
      {navigationGroups.length > 0 && <div className={styles.secondaryNavigation}>{navigationGroups.map(group => <div key={group.title}>
        <div className={styles.navLabel}>{group.title}</div><nav className={styles.navigation} aria-label={group.title}>{group.items.map(({ href, title, icon: Icon }) => <Link href={href} key={href} aria-label={title} title={title} aria-current={pathname === href ? 'page' : undefined}><Icon size={17} strokeWidth={1.5} aria-hidden="true" /><span>{title}</span>{pathname === href && <span className={styles.navMarker} />}</Link>)}</nav>
      </div>)}</div>}
      <div className={styles.sidebarBottom}>
        <div className={styles.navLabel}>RESOURCES</div>
        <nav className={styles.navigation} aria-label="Resources">
          {!account && <Link href="/dashboard" aria-label="Dashboard" title="Dashboard"><Gauge size={17} strokeWidth={1.5} aria-hidden="true" /><span>Dashboard</span></Link>}
          <Link href="/help" aria-label="Help" title="Help" aria-current={pathname === "/help" ? "page" : undefined}><CircleHelp size={17} strokeWidth={1.5} aria-hidden="true" /><span>Help & methodology</span></Link>
          <a href="/api/v1/docs" aria-label="API documentation" title="API documentation"><BookOpen size={17} strokeWidth={1.5} aria-hidden="true" /><span>API documentation</span></a>
        </nav>
        {account && <Account account={account} />}
        <div className={styles.sidebarFoot}><ShieldCheck size={14} aria-hidden="true" /><span>SPACE WEATHER<br />ANALYSIS WORKSPACE</span></div>
      </div>
    </aside>
    <div className={styles.workspaceMain}>
      <header className={styles.commandBar}>
        <div className={styles.breadcrumb}><span>ARGUS</span><ChevronRight size={12} aria-hidden="true" /><span>Space weather</span><ChevronRight size={12} aria-hidden="true" /><strong>{section}</strong></div>
        <details key={pathname} className={styles.mobileResources}>
          <summary><Menu size={13} aria-hidden="true" />Resources</summary>
          <nav aria-label="Mobile resources">
            {navigationGroups.map(group => <div key={group.title}><div className={styles.mobileGroupLabel}>{group.title}</div>{group.items.map(item => <Link key={item.href} href={item.href} aria-current={pathname === item.href ? 'page' : undefined}>{item.title}</Link>)}</div>)}
            {!account && <Link href="/dashboard">Dashboard</Link>}<Link href="/help" aria-current={pathname === "/help" ? "page" : undefined}>Help & methodology</Link><a href="/api/v1/docs">API documentation</a>
            {account && <Account account={account} />}
          </nav>
        </details>
        <div className={styles.commandStatus}><span className={styles.refreshState} data-tone={statusTone} data-warning={statusTone === 'warning'}><span className={styles.dot} />{status}</span><span className={styles.utcClock}><Clock3 size={12} aria-hidden="true" />{now ? new Date(now).toISOString().slice(11, 16) : '—'} UTC</span></div>
      </header>
      {children}
    </div>
  </div>;
}
