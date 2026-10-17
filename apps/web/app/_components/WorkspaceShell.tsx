'use client';
import DragRiskIcon from '@/components/DragRiskIcon';
import type { ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, BookOpen, ChartNoAxesCombined, ChevronRight, CircleHelp, LayoutGrid, Menu, LogOut, Radio, ShieldCheck, type LucideIcon } from 'lucide-react';
import { useDemo } from '../demo/_components/DemoContext';
import { breadcrumbs } from './breadcrumbs';
import styles from './workspace.module.css';

const navigation = [
  { href: '/', label: 'Overview', icon: LayoutGrid },
  { href: '/live', label: 'Observations', icon: Radio },
  { href: '/forecast', label: 'Forecast', icon: Activity },
  { href: '/metrics', label: 'Metrics', icon: ChartNoAxesCombined },
  { href: '/dashboard/risk/leo', label: 'Drag risk', icon: DragRiskIcon },
];

export type WorkspaceNavigationGroup = { title: string; items: { href: string; title: string; icon: LucideIcon }[] };
type WorkspaceAccount = { username: string; role: string; signingOut: boolean; logout: () => void };
function Account({ account }: { account: WorkspaceAccount }) {
  return <details className={styles.account}>
    <summary role="button" aria-label="Account menu" title={account.username}><ShieldCheck size={16} aria-hidden="true" /><span>{account.username}<small>{account.role}</small></span></summary>
    <button aria-label={account.signingOut ? 'Signing out…' : 'Sign out'} disabled={account.signingOut} onClick={account.logout}><LogOut size={14} aria-hidden="true" /><span>{account.signingOut ? 'Signing out…' : 'Sign out'}</span></button>
  </details>;
}

export default function WorkspaceShell({ children, section, contentId = 'forecast-content', navigationGroups = [], account }: {
  children: ReactNode; section: string; contentId?: string; status?: string; statusTone?: 'neutral' | 'good' | 'warning'; navigationGroups?: WorkspaceNavigationGroup[]; account?: WorkspaceAccount;
}) {
  const pathname = usePathname();
  const demo = useDemo();
  const hrefFor = (href: string) => demo ? demo.href(href) : href;
  const currentPath = demo ? pathname.replace(/^\/demo/, '') || '/' : pathname;
  const items = demo ? [
    ...navigation.filter(item => ['/', '/live', '/forecast'].includes(item.href)),
    { href: '/leo', label: 'LEO density scenario', icon: DragRiskIcon },
    { href: '/forecast/solar-events', label: 'Solar events · CME', icon: Activity },
  ] : navigation;
  return <div className={styles.workspace} data-demo={!!demo} onMouseDownCapture={event => {
    if (event.button === 0 && event.target instanceof Element && event.target.closest('.recharts-wrapper, [data-chart]')) event.preventDefault();
  }}>
    <a className={styles.skipLink} href={`#${contentId}`}>Skip to {section.toLowerCase()}</a>
    <aside className={styles.sidebar} aria-label="Workspace navigation">
      <Link className={styles.brand} href={hrefFor('/')} aria-label="Argus SunWatch home"><span>ARGUS<small>SUNWATCH</small></span></Link>
      <div className={styles.navLabel}>WORKSPACE</div>
      <nav className={styles.navigation} aria-label="Main navigation">
        {items.map(({ href, label, icon: Icon }) => {
          const active = href === '/' ? currentPath === '/' : currentPath === href || currentPath.startsWith(`${href}/`);
          return <Link href={hrefFor(href)} key={href} aria-label={label} title={label} aria-current={active ? 'page' : undefined}>
            <Icon size={17} strokeWidth={1.5} aria-hidden="true" /><span>{label}</span>{active && <span className={styles.navMarker} />}
          </Link>;
        })}
      </nav>
      {navigationGroups.length > 0 && <div className={styles.secondaryNavigation}>{navigationGroups.map(group => <div key={group.title}>
        <div className={styles.navLabel}>{group.title}</div><nav className={styles.navigation} aria-label={group.title}>{group.items.map(({ href, title, icon: Icon }) => <Link href={href} key={href} aria-label={title} title={title} aria-current={pathname === href ? 'page' : undefined}><Icon size={17} strokeWidth={1.5} aria-hidden="true" /><span>{title}</span>{pathname === href && <span className={styles.navMarker} />}</Link>)}</nav>
      </div>)}</div>}
      {!demo && <div className={styles.sidebarBottom}>
        <div className={styles.navLabel}>RESOURCES</div>
        <nav className={styles.navigation} aria-label="Resources">
          <Link href="/help" aria-label="Help" title="Help" aria-current={pathname === "/help" ? "page" : undefined}><CircleHelp size={17} strokeWidth={1.5} aria-hidden="true" /><span>Help & methodology</span></Link>
          <a href="/api/v1/docs" aria-label="API documentation" title="API documentation"><BookOpen size={17} strokeWidth={1.5} aria-hidden="true" /><span>API documentation</span></a>
        </nav>
        {account && <Account account={account} />}
      </div>}
    </aside>
    <div className={styles.workspaceMain}>
      <header className={styles.commandBar}>
        <nav className={styles.breadcrumb} aria-label="Breadcrumb">
          <ol>{breadcrumbs(currentPath, section).map((item, index) => <li key={index}>
            {index > 0 && <ChevronRight size={12} aria-hidden="true" />}
            {item.href ? <Link href={hrefFor(item.href)}>{item.label}</Link> : <span aria-current="page">{item.label}</span>}
          </li>)}</ol>
        </nav>
        {!demo && <details key={pathname} className={styles.mobileResources}>
          <summary><Menu size={13} aria-hidden="true" />Resources</summary>
          <nav aria-label="Mobile resources">
            {navigationGroups.map(group => <div key={group.title}><div className={styles.mobileGroupLabel}>{group.title}</div>{group.items.map(item => <Link key={item.href} href={item.href} aria-current={pathname === item.href ? 'page' : undefined}>{item.title}</Link>)}</div>)}
            <Link href="/help" aria-current={pathname === "/help" ? "page" : undefined}>Help & methodology</Link><a href="/api/v1/docs">API documentation</a>
            {account && <Account account={account} />}
          </nav>
        </details>}
        <Link className={styles.dashboardLink} href="/dashboard">Dashboard</Link>
      </header>
      {children}
    </div>
  </div>;
}
