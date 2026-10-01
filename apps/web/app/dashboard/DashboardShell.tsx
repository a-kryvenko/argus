'use client';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Orbit } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ApiError } from '../_utils/api';
import WorkspaceShell from '../_components/WorkspaceShell';
import { dashboardRequest, SessionContext, type User } from './session';
import { Message, Pending } from './_components/presentation';
import { dashboardLinks, navigation } from './navigation';
import './dashboard.css';

export default function DashboardShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname().replace(/\/$/, '');
  const router = useRouter();
  const login = pathname === '/dashboard/login';
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState('');
  const [signingOut, setSigningOut] = useState(false);
  useEffect(() => {
    if (login) return;
    let disposed = false;
    const refresh = () => dashboardRequest<User>('/me').then(value => {
      if (!disposed) { setUser(value); setError(''); }
    }).catch((e: unknown) => {
      if (disposed) return;
      setUser(null);
      if (e instanceof ApiError && e.status === 401) router.replace('/dashboard/login');
      else setError(e instanceof Error ? e.message : 'Could not load session');
    });
    void refresh();
    const timer = window.setInterval(refresh, 60000);
    window.addEventListener('focus', refresh);
    return () => { disposed = true; window.clearInterval(timer); window.removeEventListener('focus', refresh); };
  }, [login, pathname, router]);
  async function logout() {
    setSigningOut(true);
    try {
      await dashboardRequest('/logout', 'POST');
      setUser(null);
      router.replace('/dashboard/login');
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not sign out'); }
    finally { setSigningOut(false); }
  }
  const current = dashboardLinks.find(link => link.href === pathname);
  const account = !login && user ? { username: user.username, role: user.groups.includes('admins') ? 'Administrator' : 'Member', signingOut, logout } : undefined;
  const groups = !login && user ? navigation.map(group => ({ ...group, items: group.items.filter(item => item.href !== '/dashboard/risk/leo' && (!item.permission || user.permissions.includes(item.permission))) })).filter(group => group.items.length) : [];
  return <SessionContext value={login ? null : user}>
    <WorkspaceShell section={login ? 'Sign in' : current?.title ?? 'Dashboard'} contentId="dashboard-content" status={account?.role ?? (login ? 'Account access' : 'Opening workspace')} navigationGroups={groups} account={account}>
      <div className="dashboard dark dashboard-pane">
        <div id="dashboard-portals" />
        {login ? children : !user ? <main id="dashboard-content" className="dashboard-pending">
          <Orbit className="size-8 text-primary" />
          {error ? <><Message>{error}</Message><Button asChild variant="outline"><Link href="/dashboard/login">Return to sign in</Link></Button></> : <Pending label="Opening your workspace…" />}
        </main> : <main id="dashboard-content" className="dashboard-content">
          {error && <Message>{error}</Message>}
          {current?.permission && !user.permissions.includes(current.permission) ? <Message>You do not have access to this section.</Message> : children}
          <footer className="dashboard-footer"><span>ARGUS Sunwatch · {account?.role} workspace</span><span>Space weather · all times UTC</span></footer>
        </main>}
      </div>
    </WorkspaceShell>
  </SessionContext>;
}
