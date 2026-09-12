"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import "./header.css";

const links = [
  ["/", "Forecast"],
  ["/live", "Live"],
  ["/products", "Products"],
  ["/metrics", "Metrics"],
  ["/help", "Help"],
  ["/dashboard", "Dashboard"],
];

export default function Header() {
  const pathname = usePathname();
  return (
    <header className="header">
      <div className="container header__inner">
        <Link
          className="header__brand"
          href="/"
          aria-label="Argus SunWatch home"
        >
          <span>Argus SunWatch</span>
        </Link>
        <nav className="header__nav" aria-label="Main navigation">
          {links.map(([href, label]) => {
            const active =
              href === "/"
                ? pathname === href
                : pathname === href || pathname.startsWith(`${href}/`);
            return (
              <Link
                href={href}
                key={href}
                aria-current={active ? "page" : undefined}
              >
                {label}
              </Link>
            );
          })}
          <a href="/api/v1/docs">API docs</a>
        </nav>
      </div>
    </header>
  );
}
