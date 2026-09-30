"use client";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Activity, ArrowDown, Gauge, Layers3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { apiRequest, ApiError } from "../../_utils/api";
import { EmptyState, Message, MetricCard, PageHeading, Pending, selectClass } from "./presentation";
import { type DragAssessment, type DragInputs, number, utc } from "./leo-drag";

const Chart = dynamic(() => import("./LeoDragChart"), {
  ssr: false,
  loading: () => <div className="flex h-[280px] items-center justify-center"><Pending label="Loading chart…" /></div>,
});
const fields = [
  { name: "altitude_km", label: "Altitude (km)", min: 200, max: 800, value: 400 },
  { name: "inclination_deg", label: "Inclination (°)", min: 0, max: 180, value: 51.6 },
  { name: "mass_kg", label: "Mass (kg)", min: 0.01, max: 1e7, value: 100 },
  { name: "effective_area_m2", label: "Effective area (m²)", min: 0, max: 1e6, value: 1 },
  { name: "drag_coefficient", label: "Drag coefficient (Cd)", min: 0, max: 10, value: 2.2 },
] as const;

/** Standalone assessment view; routing and access policy belong to the dashboard shell. */
export default function LeoDragView() {
  const [result, setResult] = useState<DragAssessment | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);

  async function calculate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const inputs = Object.fromEntries([...form].map(([key, value]) => [key, Number(value)])) as DragInputs;
    if (inputs.effective_area_m2 <= 0 || inputs.drag_coefficient <= 0) {
      setError("Effective area and drag coefficient must be greater than zero.");
      return;
    }
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const data = await apiRequest<DragAssessment>("/public/risks/leo-drag?meta=true", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs), cache: "no-store", signal: controller.signal,
      });
      if (!controller.signal.aborted) setResult(data);
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(e instanceof ApiError && e.status === 503
        ? "Drag assessment is unavailable. Fresh atmospheric density data or the calculation service is not ready. Try again later."
        : e instanceof ApiError && e.status === 422
          ? "These parameters are outside the available density grid or model limits. Check the inputs; estimated decay must not exceed 1,000 m."
          : "Could not calculate drag. Please try again.");
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return (
    <>
      <PageHeading title="LEO drag assessment" description="Estimate atmospheric drag and altitude loss for a circular low Earth orbit." />
      <div className="grid items-start gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
        <Card className="p-5">
          <h2 className="font-medium">Orbit & spacecraft</h2>
          <p className="mt-2 text-xs text-muted-foreground">Illustrative inputs. Adjust them for your spacecraft and attitude.</p>
          <form onSubmit={calculate} className="mt-5 space-y-4">
            <fieldset disabled={busy} className="space-y-4">
              {fields.map(field => (
                <div key={field.name} className="space-y-2">
                  <Label htmlFor={`leo-${field.name}`}>{field.label}</Label>
                  <Input id={`leo-${field.name}`} name={field.name} type="number" required step="any"
                    min={field.min} max={field.max} defaultValue={field.value} />
                </div>
              ))}
              <div className="space-y-2">
                <Label htmlFor="leo-horizon">Assessment horizon</Label>
                <select id="leo-horizon" name="horizon_hours" defaultValue="24" className={selectClass}>
                  <option value="24">24 hours</option><option value="48">48 hours</option>
                </select>
              </div>
              <p className="text-xs text-muted-foreground">Starts at the density forecast issue time, not the time of this request.</p>
              <Button type="submit" className="w-full" disabled={busy}>{busy ? "Calculating…" : "Calculate drag"}</Button>
            </fieldset>
          </form>
        </Card>
        <div className="min-w-0 space-y-6" aria-busy={busy}>
          <Card className="p-5 text-sm">
            <p className="font-medium">Physical estimates · Risk rules pending</p>
            <p className="mt-2 text-muted-foreground">Risk categories are not assigned in this panel yet. Calculations hold current atmospheric drivers constant and do not predict future storms.</p>
          </Card>
          {error && <Message>{error}</Message>}
          {busy && <Pending label="Calculating orbital drag…" />}
          {!result && !busy && <Card><EmptyState title="Ready to calculate" description="Enter orbit and spacecraft parameters to see drag estimates and an hourly altitude-loss profile." /></Card>}
          {result && <>
            <section aria-label="Assessment results" className="space-y-4">
              <div className="text-sm text-muted-foreground">
                <p>{utc(result.start_time)} – {utc(result.end_time)} UTC</p>
                <p className="mt-1">Calculated for {result.inputs.altitude_km} km · {result.inputs.inclination_deg}° · {result.inputs.mass_kg} kg · {result.inputs.effective_area_m2} m² · Cd {result.inputs.drag_coefficient}</p>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <MetricCard label="Estimated altitude loss" value={`${number(result.estimated_altitude_loss_m)} m`} detail={`Cumulative over ${result.inputs.horizon_hours} hours`} icon={<ArrowDown className="size-4" />} />
                <MetricCard label="Drag Δv" value={`${number(result.delta_v_loss_m_s)} m/s`} detail="Accumulated along-track drag impulse" icon={<Gauge className="size-4" />} />
                <MetricCard label="Mean density" value={`${number(result.mean_density_kg_m3)} kg/m³`} detail="Averaged over orbit and time" icon={<Layers3 className="size-4" />} />
                <MetricCard label="Mean drag acceleration" value={`${number(result.mean_drag_accel_m_s2)} m/s²`} detail="Acceleration magnitude" icon={<Activity className="size-4" />} />
              </div>
            </section>
            <Card className="min-w-0 p-5"><h2 className="mb-4 font-medium">Altitude-loss profile</h2><Chart rows={result.predictions} /></Card>
            <Card className="min-w-0 p-5">
              <details><summary className="cursor-pointer font-medium">Hourly estimates</summary>
                <Table className="mt-4"><TableHeader><TableRow>
                  <TableHead>Lead (h)</TableHead><TableHead>Time (UTC)</TableHead><TableHead>Density (kg/m³)</TableHead><TableHead>Drag Δv (m/s)</TableHead><TableHead>Altitude loss (m)</TableHead>
                </TableRow></TableHeader><TableBody>{result.predictions.map(point => <TableRow key={point.lead_hours}>
                  <TableCell>{point.lead_hours}</TableCell><TableCell className="whitespace-nowrap">{utc(point.valid_time)}</TableCell><TableCell>{number(point.mean_density_kg_m3)}</TableCell><TableCell>{number(point.delta_v_loss_m_s)}</TableCell><TableCell>{number(point.estimated_altitude_loss_m)}</TableCell>
                </TableRow>)}</TableBody></Table>
              </details>
            </Card>
            <Card className="space-y-3 p-5 text-sm">
              <h2 className="font-medium">Data & assumptions</h2>
              <dl className="grid gap-2 text-muted-foreground sm:grid-cols-2">
                {result.meta && <div><dt>Density model</dt><dd>{result.meta.source.model}</dd></div>}
                <div><dt>Forecast issued (UTC)</dt><dd>{utc(result.source.issue_time)}</dd></div>
                <div><dt>Solar drivers observed (UTC)</dt><dd>{utc(result.source.observed_at)}</dd></div>
                <div><dt>DTC observed (UTC)</dt><dd>{utc(result.source.dtc_observed_at)}</dd></div>
              </dl>
              {result.source.background_interpolated && <p className="text-xs text-muted-foreground">Background drivers include interpolated daily values.</p>}
              <ul className="list-disc space-y-2 pl-5 text-xs text-muted-foreground">{result.assumptions.map(item => <li key={item}>{item}</li>)}</ul>
            </Card>
          </>}
        </div>
      </div>
    </>
  );
}
