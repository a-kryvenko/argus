import DragRiskIcon from '@/components/DragRiskIcon';
import Link from 'next/link';
import { ArrowUpRight, Activity, ChartNoAxesCombined, Radio } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { PageHeading } from './presentation';

const sections = [
  { href: '/live', title: 'Live observations', description: 'Follow solar wind and geomagnetic conditions with shared UTC history controls.', icon: Radio },
  { href: '/forecast', title: 'Forecast', description: 'Explore model outlooks, prediction intervals and threshold probabilities.', icon: Activity },
  { href: '/metrics', title: 'Model performance', description: 'Inspect forecast quality by lead hour and probability calibration.', icon: ChartNoAxesCombined },
  { href: '/dashboard/risk/leo', title: 'Drag risk', description: 'Monitor atmospheric density risk for your spacecraft.', icon: DragRiskIcon },
];
export default function ClientOverview() {
  return <>
    <PageHeading title="Your workspace" description="Observations, forecasts and risk assessments in one workspace." />
    <div className="grid gap-4 md:grid-cols-2">{sections.map(({ href, title, description, icon: Icon }) => <Card key={href} className="shadow-none transition-colors hover:border-primary/50">
      <Link href={href} className="flex h-full flex-col items-start p-6">
        <Icon className="mb-5 size-5 text-primary" aria-hidden="true" />
        <h2 className="font-medium">{title}</h2><p className="mt-2 text-sm leading-relaxed text-muted-foreground">{description}</p>
        <span className="mt-6 flex w-full items-center justify-between border-t pt-4 text-xs text-primary">Open section<ArrowUpRight className="size-4" aria-hidden="true" /></span>
      </Link>
    </Card>)}</div>
  </>;
}
