'use client';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import DragMonitorView from './DragMonitorView';
import LeoDragView from './LeoDragView';
import DensityAdvisoryView from './DensityAdvisoryView';

export default function LeoRiskView() {
  const [view, setView] = useState<'density' | 'monitor' | 'calculator'>('density');
  return <>
    <nav aria-label="Drag risk views" className="flex flex-wrap gap-2">
      <Button variant={view === 'density' ? 'default' : 'outline'} aria-pressed={view === 'density'} onClick={() => setView('density')}>Density advisories</Button>
      <Button variant={view === 'monitor' ? 'default' : 'outline'} aria-pressed={view === 'monitor'} onClick={() => setView('monitor')}>Experimental loss monitoring</Button>
      <Button variant={view === 'calculator' ? 'default' : 'outline'} aria-pressed={view === 'calculator'} onClick={() => setView('calculator')}>Scenario calculator</Button>
    </nav>
    {view === 'density' ? <DensityAdvisoryView /> : view === 'monitor' ? <DragMonitorView /> : <><p className="text-sm text-muted-foreground">Experimental altitude-loss estimate. Flight-loss accuracy is unvalidated; use density advisories for operator attention.</p><LeoDragView /></>}
  </>;
}
