'use client';

import { usePathname } from 'next/navigation';
import Header from './Header';
import Footer from './Footer';

export default function SiteChrome({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  if (pathname === '/dashboard' || pathname.startsWith('/dashboard/')) return <>{children}</>;
  return <><div><Header />{children}</div><Footer /></>;
}
