'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import './header.css'

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="header color-default">
      <div className="container d-flex justify-content-between">
        <a href="/">A</a>
        <div className="d-flex justify-content-end">
          <a href="/">Forecast</a>
          <span>&nbsp;|&nbsp;<a href="/metrics">Metrics</a></span>
          <span>&nbsp;|&nbsp;<a href="/api/v1/docs">API</a></span>
          <span>&nbsp;|&nbsp;<a href="/help">Help</a></span>
        </div>
      </div>
    </header>
  );
}