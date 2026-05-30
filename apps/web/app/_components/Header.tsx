'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import './header.css'

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="header">
      <div className="container d-flex justify-content-center">
        <h1 className="header__title"><a href="/">Argus SunWatch</a> - SOLAR IMPACT INTELLIGENCE</h1>
        <span>&nbsp;|&nbsp;<a href="/help">Help?</a></span>
      </div>
    </header>
  );
}