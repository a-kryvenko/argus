'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-cyan-500 rounded-full flex items-center justify-center text-black font-bold text-xl">
            A
          </div>
          <div>
            <div className="font-semibold text-xl tracking-tight">Argus SunWatch</div>
            <div className="text-[10px] text-zinc-500 -mt-1">SOLAR IMPACT INTELLIGENCE</div>
          </div>
        </div>

        <nav className="flex items-center gap-8 text-sm">
          <Link 
            href="/" 
            className={`hover:text-white transition-colors ${pathname === '/' ? 'text-white font-medium' : 'text-zinc-400'}`}
          >
            Public Forecast
          </Link>
          
          <Link 
            href="/dashboard" 
            className={`hover:text-white transition-colors ${pathname === '/dashboard' ? 'text-white font-medium' : 'text-zinc-400'}`}
          >
            Private Dashboard
          </Link>
        </nav>

        <div className="text-xs text-zinc-500 font-mono">
          Surya-1.0
        </div>
      </div>
    </header>
  );
}