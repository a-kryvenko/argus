'use client';
import { useEffect, useState } from 'react';
import { apiRequest } from '../_utils/api';
import type { Latest } from './solarWind';
import type { IndexLatest } from './geomagnetic';

export type Derived = { status: 'available' | 'lower_bound' | 'unavailable'; value: number | null; reason?: string; as_of?: string };
export type Summary = {
  generated_at: string;
  solar_wind: Latest['series'];
  geomagnetic: IndexLatest['series'];
  changes_1h: Record<string, Derived>;
  southward_bz: Derived;
};
export type QueryResult<T> = { data?: T; receivedAt?: number; failed: boolean; loading: boolean };
export type LiveSnapshot = QueryResult<Summary> & { now?: number };
export type HistoryWindow = { from: string; to: string };

// One request at a time, keep the previous response on failure, discard responses
// from an old period. History views receive one shared UTC window from the page.
export function useLiveQuery<T>(path: string | null, range?: HistoryWindow): QueryResult<T> {
  const from = range?.from, to = range?.to;
  // Keep the last successful response while the same period advances, but never
  // display a previous 24-hour response under a newly selected 3-day period.
  const key = `${path}:${from && to ? Date.parse(to) - Date.parse(from) : ''}`;
  const [state, setState] = useState<QueryResult<T> & { key: string }>({ key, failed: false, loading: true });
  useEffect(() => {
    if (!path) return;
    const queryPath = path;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      const url = new URL(queryPath, 'http://local');
      if (from && to) {
        url.searchParams.set('from', from);
        url.searchParams.set('to', to);
      }
      try {
        const data = await apiRequest<T>(url.pathname+url.search, { signal: controller.signal, cache: 'no-store' });
        if (controller.signal.aborted) return;
        setState({ key, data, receivedAt: Date.now(), failed: false, loading: false });
      } catch {
        if (controller.signal.aborted) return;
        setState(previous => ({ ...(previous.key === key ? previous : {}), key, failed: true, loading: false }));
      }
      // The page advances history windows once a minute, also after failed reads.
      if (!controller.signal.aborted && !from) timer = setTimeout(refresh, 60000);
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [path, from, to, key]);
  return state.key === key ? state : { failed: false, loading: true };
}

export function useLiveClock() {
  const [now, setNow] = useState<number>();
  useEffect(() => {
    const initial = setTimeout(() => setNow(Date.now()), 0);
    const timer = setInterval(() => setNow(Date.now()), 30000);
    return () => { clearTimeout(initial); clearInterval(timer); };
  }, []);
  return now;
}
