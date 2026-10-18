'use client';
import { createContext, useContext } from 'react';
import type { DemoBundle } from '../_lib/replay';

export const DemoContext = createContext<{
  bundle: DemoBundle; now: number; offset: number; showFuture: boolean; href: (path: string) => string;
} | null>(null);
export const useDemo = () => useContext(DemoContext);
