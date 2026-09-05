'use client';

import Link from 'next/link';
import './header.css'

export default function Header() {
  return (
    <header className="header color-default">
      <div className="container d-flex justify-content-between">
        <a href="/">A</a>
        <nav className="header__nav" aria-label="Main navigation">
          <a href="/">Forecast</a>
          <span>&nbsp;|&nbsp;<Link href="/live">Live</Link></span>
          <span>&nbsp;|&nbsp;<a href="/products">Products</a></span>
          <span>&nbsp;|&nbsp;<a href="/metrics">Metrics</a></span>
          <span>&nbsp;|&nbsp;<a href="/api/v1/docs">API docs</a></span>
          <span>&nbsp;|&nbsp;<a href="/help">Help</a></span>
        </nav>
      </div>
    </header>
  );
}
