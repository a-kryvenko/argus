'use client';
import { useEffect, useState } from 'react';
import { apiRequest } from '../_utils/api';
import type { Latest } from './solarWind';
import type { IndexLatest } from './geomagnetic';

export type Derived = { status: 'available' | 'lower_bound' | 'unavailable'; value: number | null; reason?: string; as_of?: string; unit?: string };
export type Summary = {
  generated_at: string;
  solar_wind: Latest['series'];
  geomagnetic: IndexLatest['series'];
  changes_1h: Record<string, Derived>;
  southward_bz: Derived;
};
export type QueryResult<T> = { data?: T; receivedAt?: number; failed: boolean; loading: boolean };
export type LiveSnapshot = QueryResult<Summary> & { now?: number };

// One request at a time, keep the previous response on failure, discard responses
// from an old period. The range is rebuilt on each poll so history keeps moving.
export function useLiveQuery<T>(path: string, hours?: number): QueryResult<T> {
  const key = `${path}:${hours ?? ''}`;
  const [state, setState] = useState<QueryResult<T> & { key: string }>({ key, failed: false, loading: true });
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      const url = new URL(path, 'http://local');
      if (hours !== undefined) {
        const end = new Date();
        url.searchParams.set('from', new Date(end.getTime()-hours*3600000).toISOString());
        url.searchParams.set('to', end.toISOString());
      }
      try {
        const data = await apiRequest<T>(url.pathname+url.search, { signal: controller.signal, cache: 'no-store' });
        if (controller.signal.aborted) return;
        setState({ key, data, receivedAt: Date.now(), failed: false, loading: false });
      } catch {
        if (controller.signal.aborted) return;
        setState(previous => ({ ...(previous.key === key ? previous : {}), key, failed: true, loading: false }));
      }
      if (!controller.signal.aborted) timer = setTimeout(refresh, 60000);
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [path, hours, key]);
  return state.key === key ? state : { failed: false, loading: true };
}

export function useLiveClock() {
  const [now, setNow] = useState<number>();
  useEffect(() => {
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 30000);
    return () => clearInterval(timer);
  }, []);
  return now;
}
