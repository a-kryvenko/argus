"use client";
import { useEffect, useState } from "react";
import { Badge } from "@/app/dashboard/_components/ui/badge";
import { Card } from "@/app/dashboard/_components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/dashboard/_components/ui/table";
import { dashboardRequest } from "../session";
import {
  Message,
  PageHeading,
  Pending,
} from "./presentation";

type Measurement = {
  metric: string;
  latest_observation_at: string | null;
  received_at: string | null;
  status: string;
  stale_after_seconds: number;
};
type Forecast = {
  product: string;
  status: string;
  freshness: string;
  error?: string;
  current_release: { issue_time: string; published_at: string } | null;
  latest_attempt: {
    status: string;
    started_at: string;
    finished_at: string | null;
    error: string | null;
  } | null;
  latest_attempt_artifacts: {
    name: string;
    status: string;
    error: string | null;
  }[];
};
type Project = {
  status: string;
  checked_at: string | null;
  stale: boolean;
  observations: {
    status: string;
    measurements: Measurement[];
    error?: string;
  } | null;
  forecasts: Forecast[];
};
const labels: Record<string, string> = {
  ok: "Healthy",
  fresh: "Fresh",
  down: "Unavailable",
  unknown: "Unknown",
  warning: "Needs attention",
  degraded: "Needs attention",
  unavailable: "Unavailable",
  delayed: "Delayed",
  stale: "Outdated",
  within_age_limit: "Fresh",
  future_issue_time: "Future issue time",
  unconfigured: "Freshness limit not set",
  future: "Future timestamp",
  running: "Running",
  succeeded: "Completed",
  failed: "Failed",
  partial: "Partial",
  interrupted: "Interrupted",
};
const productLabels: Record<string, string> = {
  "solar-wind-speed": "Solar wind speed",
  "solar-wind-density": "Solar wind density",
  hmf: "Magnetic field",
  "solar-radiation": "Solar radiation",
  "geomagnetic-activity": "Geomagnetic activity",
  dst: "Dst",
  "atmospheric-density": "Atmospheric density",
};
function Status({ value }: { value: string }) {
  const good = ["ok", "fresh", "within_age_limit", "succeeded"].includes(value);
  return (
    <Badge
      variant="outline"
      className={
        good
          ? "border-emerald-500/30 text-emerald-300"
          : "border-amber-500/30 text-amber-200"
      }
    >
      {labels[value] ?? value.replaceAll("_", " ")}
    </Badge>
  );
}
function date(value: string | null | undefined) {
  return value
    ? new Date(value).toISOString().replace("T", " ").slice(0, 19)
    : "—";
}
function Stamp({ value }: { value: string | null | undefined }) {
  return (
    <time
      dateTime={value ?? undefined}
      className="whitespace-nowrap tabular-nums"
    >
      {date(value)}
    </time>
  );
}
export default function ProjectOverview() {
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState("");
  const [clock, setClock] = useState(() => Date.now());
  useEffect(() => {
    let disposed = false;
    async function refresh() {
      try {
        const result = await dashboardRequest<Project>("/project-monitoring");
        if (disposed) return;
        setProject(result);
        setError("");
      } catch {
        if (disposed) return;
        setError("Could not refresh data status.");
      }
      setClock(Date.now());
    }
    void refresh();
    const timer = window.setInterval(refresh, 30000);
    const tick = window.setInterval(() => setClock(Date.now()), 10000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      window.clearInterval(tick);
    };
  }, []);
  const old =
    !project?.checked_at ||
    clock - Date.parse(project.checked_at) > 120000 ||
    project.stale ||
    !!error;
  const healthy = !old && project?.status === "ok";
  return (
    <>
      <PageHeading
        title="Project overview"
        description="Observation freshness and forecasts. All times are UTC."
      />
      <Card className="flex flex-wrap items-center justify-between gap-3 p-5">
        <div>
          <p className="font-medium">
            {healthy
              ? "Observations and forecasts are current"
              : old
                ? "No current data status"
                : "Observations or forecasts need attention"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Last check: <Stamp value={project?.checked_at} /> · Automatic checks
            every 30 seconds
          </p>
        </div>
        <Status value={healthy ? "ok" : old ? "unknown" : "degraded"} />
      </Card>
      {error && (
        <Message>
          {error} Previously loaded values may be outdated.
        </Message>
      )}
      {!project && !error && (
        <Pending label="Checking observations and forecasts…" />
      )}
      {project && old && (
        <Message>
          Checks are missing or older than two minutes. Values may be outdated.
        </Message>
      )}
      {project && (
        <>
          <section className="space-y-3">
            <h2 className="text-lg font-medium">
              Clio · Observations
            </h2>
            {project.observations?.error && (
              <Message>{project.observations.error}</Message>
            )}
            <Card className="overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Latest observation</TableHead>
                    <TableHead>Last received</TableHead>
                    <TableHead>Freshness limit</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(project.observations?.measurements ?? []).map((item) => (
                    <TableRow key={item.metric}>
                      <TableCell className="font-medium">
                        {item.metric}
                      </TableCell>
                      <TableCell>
                        <Status value={old ? "unknown" : item.status} />
                      </TableCell>
                      <TableCell>
                        <Stamp value={item.latest_observation_at} />
                      </TableCell>
                      <TableCell>
                        <Stamp value={item.received_at} />
                      </TableCell>
                      <TableCell>{item.stale_after_seconds / 3600} h</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </section>
          <section className="space-y-3">
            <h2 className="text-lg font-medium">Prophet · Forecast</h2>
            <Card className="overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Product</TableHead>
                    <TableHead>Freshness</TableHead>
                    <TableHead>Issue time</TableHead>
                    <TableHead>Published</TableHead>
                    <TableHead>Latest generation</TableHead>
                    <TableHead>Started / finished</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {project.forecasts.map((item) => (
                    <TableRow key={item.product}>
                      <TableCell>
                        <p className="font-medium">
                          {productLabels[item.product] ?? item.product}
                        </p>
                        {(item.error || item.latest_attempt?.error) && (
                          <p className="mt-1 max-w-60 text-xs text-amber-200">
                            {item.error || item.latest_attempt?.error}
                          </p>
                        )}
                      </TableCell>
                      <TableCell>
                        <Status value={old ? "unknown" : item.freshness} />
                      </TableCell>
                      <TableCell>
                        <Stamp value={item.current_release?.issue_time} />
                      </TableCell>
                      <TableCell>
                        <Stamp value={item.current_release?.published_at} />
                      </TableCell>
                      <TableCell>
                        <Status
                          value={
                            old
                              ? "unknown"
                              : (item.latest_attempt?.status ?? "unknown")
                          }
                        />
                      </TableCell>
                      <TableCell>
                        <Stamp value={item.latest_attempt?.started_at} />
                        <br />
                        <Stamp value={item.latest_attempt?.finished_at} />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
            <p className="text-xs text-muted-foreground">
              The published forecast remains visible if a later generation
              fails. Freshness does not establish scientific model quality.
            </p>
          </section>
        </>
      )}
    </>
  );
}
