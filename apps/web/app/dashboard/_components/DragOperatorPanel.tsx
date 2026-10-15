'use client';
import dynamic from 'next/dynamic';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Pending } from './presentation';
import { dataLabels, type OperatorChart } from './drag-monitor';
import { number, utc } from './leo-drag';

const Chart = dynamic(() => import('./DragOperatorChart'), { ssr: false, loading: () => <Pending label="Loading chart…" /> });

export default function DragOperatorPanel({ chart }: { chart: OperatorChart }) {
  const [showPrevious, setShowPrevious] = useState(true);
  return <section aria-label="Operator drag chart" className="min-w-0 space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h3 className="font-medium">Attitude modes & limits</h3>
      <Button size="sm" variant="outline" aria-pressed={showPrevious} onClick={() => setShowPrevious(!showPrevious)}>{showPrevious ? 'Hide previous releases' : 'Show previous releases'}</Button>
    </div>
    <p className={chart.data_status === 'current' ? 'text-sm' : 'text-sm text-amber-300'}>{chart.historical ? 'Historical snapshot' : 'Latest snapshot'} · {dataLabels[chart.data_status]}</p>
    {chart.start_time && <p className="text-xs text-muted-foreground">{utc(chart.start_time)} – {chart.end_time && utc(chart.end_time)} UTC</p>}
    <p className="text-xs text-muted-foreground">Now: {utc(chart.prepared_at)} UTC · {chart.now_position === 'within' ? 'Within the selected interval' : chart.now_position === 'before' ? 'Before the selected interval' : chart.now_position === 'after' ? 'Selected interval has ended' : 'No calculable interval'}</p>
    {chart.modes.some(mode => mode.points.length > 0) && <Chart chart={chart} showPrevious={showPrevious} />}
    <p className="text-xs text-muted-foreground">Shading and faded lines mark elapsed time. All curves are modeled scenarios, including elapsed portions. Dashed curves compare previous releases at equal lead hours; their UTC dates differ. Diamonds mark the first hourly samples at or above a limit.</p>
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{chart.modes.map(mode => <div key={mode.mode} className="space-y-2 rounded-lg border p-3 text-sm">
      <h4 className="font-medium">{mode.mode}{mode.active && ' · Active'}</h4>
      {mode.error ? <p className="text-amber-300">{mode.error === 'outside_model_domain' ? 'Outside model limits · No curve available' : 'Assessment unavailable · No curve available'}</p> : <>
        {mode.crossings.length === 0 && <p>No limits reached in this scenario.</p>}
        {mode.crossings.map(crossing => <p key={crossing.level}>{crossing.level === 'high' ? 'High' : 'Review'}: {utc(crossing.valid_time)} UTC · {crossing.state === 'elapsed' ? 'Elapsed' : 'Upcoming'}</p>)}
        {mode.loss_change_m !== null ? <p>Change: {mode.loss_change_m > 0 ? '+' : ''}{number(mode.loss_change_m)} m<br /><span className="text-xs text-muted-foreground">Compared with {mode.previous_issue_time && utc(mode.previous_issue_time)} UTC</span></p> : <p className="text-xs text-muted-foreground">No comparable previous release.</p>}
      </>}
    </div>)}</div>
    <details className="text-sm"><summary className="cursor-pointer">Chart data</summary>
      <Table className="mt-3"><TableHeader><TableRow><TableHead>Mode</TableHead><TableHead>Release</TableHead><TableHead>Lead (h)</TableHead><TableHead>Time (UTC)</TableHead><TableHead>Altitude loss (m)</TableHead></TableRow></TableHeader>
        <TableBody>{chart.modes.flatMap(mode => (showPrevious ? [['Selected', mode.points], ['Previous', mode.previous]] as const : [['Selected', mode.points]] as const).flatMap(([label, points]) => points.map(point => <TableRow key={`${mode.mode}-${label}-${point.lead_hours}`}><TableCell>{mode.mode}</TableCell><TableCell>{label}</TableCell><TableCell>{number(point.lead_hours)}</TableCell><TableCell className="whitespace-nowrap">{utc(point.valid_time)}</TableCell><TableCell>{number(point.altitude_loss_m)}</TableCell></TableRow>)))}</TableBody>
      </Table>
    </details>
  </section>;
}
