'use client';
import { useEffect, useState } from 'react';

export function useClock() {
  const [now, setNow] = useState<number>();
  useEffect(() => {
    const initial = setTimeout(() => setNow(Date.now()), 0);
    const timer = setInterval(() => setNow(Date.now()), 30000);
    return () => { clearTimeout(initial); clearInterval(timer); };
  }, []);
  return now;
}
