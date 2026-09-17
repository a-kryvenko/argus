"use client";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Activity,
  Cpu,
  Database,
  HardDrive,
  RefreshCw,
  Server,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { dashboardRequest, useSession } from "../session";
import {
  Message,
  MetricCard,
  PageHeading,
  Pending,
  selectClass,
} from "./presentation";
import type { Summary } from "./ApiActivity";

const TrafficChart = dynamic(() => import("./TrafficChart"), { ssr: false });
type Service = {
  name: string;
  status: string;
  detail?: string;
  response_ms?: number;
};
type Source = {
  source_id: string;
  label: string;
  status: string;
  latest_observation_at: string | null;
  latest_interval_end: string | null;
  last_response_at: string | null;
  last_success_at: string | null;
  last_attempt_at: string | null;
  last_error_message: string | null;
  consecutive_failures: number;
};
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
  services: Service[];
  observations: {
    status: string;
    sources: Record<string, Source>;
    measurements: Measurement[];
    error?: string;
    last_refresh_completed_at?: string | null;
  } | null;
  forecasts: Forecast[];
  host: {
    status: string;
    cpu_percent: number | null;
    memory_percent: number | null;
    disk_percent: number | null;
    disk_free_bytes: number | null;
  } | null;
};
type Traffic = {
  recent_errors_5xx: number;
  status: string;
  checked_at: string | null;
  stale: boolean;
  since: string | null;
  resolution: string;
  channels: Record<
    "api" | "site",
    { summary: Summary; points: (Summary & { time: string })[] }
  > | null;
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
  collecting: "Collecting",
  source_delayed: "Source delayed",
  collection_error: "Collection failed",
  collector_stalled: "Collector stalled",
  collector_overdue: "Collector overdue",
  data_unavailable: "No usable data",
  data_partial: "Partial data",
  not_started: "Not started",
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
function percent(value: number | null | undefined) {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

export default function ProjectOverview() {
  const user = useSession();
  const [project, setProject] = useState<Project | null>(null);
  const [traffic, setTraffic] = useState<Traffic | null>(null);
  const [period, setPeriod] = useState("day");
  const [version, setVersion] = useState(0);
  const [errors, setErrors] = useState({ project: "", traffic: "" });
  const [clock, setClock] = useState(() => Date.now());
  useEffect(() => {
    let disposed = false;
    async function refresh() {
      const results = await Promise.allSettled([
        dashboardRequest<Project>("/project-monitoring"),
        dashboardRequest<Traffic>(`/project-traffic?period=${period}`),
      ]);
      if (disposed) return;
      setClock(Date.now());
      if (results[0].status === "fulfilled") setProject(results[0].value);
      if (results[1].status === "fulfilled") setTraffic(results[1].value);
      setErrors({
        project:
          results[0].status === "rejected"
            ? "Could not refresh project status."
            : "",
        traffic:
          results[1].status === "rejected"
            ? "Could not refresh traffic statistics."
            : "",
      });
    }
    void refresh();
    const timer = window.setInterval(refresh, 30000);
    const tick = window.setInterval(() => setClock(Date.now()), 10000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      window.clearInterval(tick);
    };
  }, [period, version]);
  const old =
    !project?.checked_at ||
    clock - Date.parse(project.checked_at) > 120000 ||
    project.stale ||
    !!errors.project;
  const trafficOld =
    !traffic?.checked_at ||
    clock - Date.parse(traffic.checked_at) > 120000 ||
    traffic.stale ||
    !!errors.traffic;
  const recentErrors = Number(traffic?.recent_errors_5xx ?? 0);
  const healthy =
    !old &&
    project?.status === "ok" &&
    !trafficOld &&
    traffic?.status === "ok" &&
    recentErrors === 0;
  return (
    <>
      <PageHeading
        title="Project overview"
        description="Service availability, observation freshness, forecasts and traffic. All times are UTC."
        action={
          <Button variant="outline" onClick={() => setVersion((v) => v + 1)}>
            <RefreshCw className="mr-2 size-4" />
            Refresh
          </Button>
        }
      />
      <Card className="flex flex-wrap items-center justify-between gap-3 p-5">
        <div>
          <p className="font-medium">
            {healthy
              ? "All systems operational"
              : old
                ? "No current project status"
                : "Project needs attention"}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Last check: <Stamp value={project?.checked_at} /> · Automatic checks
            every 30 seconds
          </p>
        </div>
        <Status value={healthy ? "ok" : old ? "unknown" : "degraded"} />
      </Card>
      {recentErrors > 0 && (
        <Message>
          {recentErrors} server errors (5xx) in the last five minute buckets.
          Check API and website traffic below.
        </Message>
      )}
      {errors.project && (
        <Message>
          {errors.project} Previously loaded values may be outdated.
        </Message>
      )}
      {!project && !errors.project && (
        <Pending label="Checking project status…" />
      )}
      {project && (
        <>
          {old && (
            <Message>
              Checks are missing or older than two minutes. The values below do
              not confirm current availability.
            </Message>
          )}
          <section
            aria-label="Services"
            className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3"
          >
            {project.services.map((service) => (
              <Card key={service.name} className="p-4">
                <div className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-sm font-medium">
                    <Server className="size-4 text-muted-foreground" />
                    {service.name}
                  </span>
                  <Status value={old ? "unknown" : service.status} />
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  {service.detail ??
                    (service.response_ms != null
                      ? `${service.response_ms} ms`
                      : "Connection available")}
                </p>
              </Card>
            ))}
          </section>
          <section
            aria-label="Server resources"
            className="grid gap-4 sm:grid-cols-3"
          >
            <MetricCard
              label="CPU"
              value={percent(project.host?.cpu_percent)}
              detail={old ? "Last known sample" : "Host CPU between checks"}
              icon={<Cpu className="size-4" />}
            />
            <MetricCard
              label="Memory"
              value={percent(project.host?.memory_percent)}
              detail="Host memory in use"
              icon={<Database className="size-4" />}
            />
            <MetricCard
              label="Disk"
              value={percent(project.host?.disk_percent)}
              detail={
                project.host?.disk_free_bytes != null
                  ? `${(project.host.disk_free_bytes / 1024 ** 3).toFixed(1)} GiB available on the logs filesystem`
                  : "Host metrics are not available"
              }
              icon={<HardDrive className="size-4" />}
            />
          </section>
        </>
      )}
      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-medium">
            Traffic · API and public website
          </h2>
          <select
            className={`${selectClass} max-w-48`}
            aria-label="Traffic period"
            value={period}
            onChange={(event) => {
              setTraffic(null);
              setPeriod(event.target.value);
            }}
          >
            <option value="hour">Last hour · minutes</option>
            <option value="day">Last 24 hours</option>
            <option value="week">Last 7 days</option>
          </select>
        </div>
        {errors.traffic && <Message>{errors.traffic}</Message>}
        {(trafficOld || traffic?.status !== "ok") && (
          <Message>
            Traffic collection is unavailable, catching up or outdated. Counts
            may be incomplete.
          </Message>
        )}
        <p className="text-xs text-muted-foreground">
          Public page requests include bots. Static assets, dashboard pages and
          monitoring probes are excluded. Collection started:{" "}
          <Stamp value={traffic?.since} />.
        </p>
        {traffic?.channels && (
          <div className="grid gap-4 xl:grid-cols-2">
            {(["api", "site"] as const).map((channel) => {
              const data = traffic.channels![channel];
              const start = traffic.since ? Date.parse(traffic.since) : 0;
              const rows = data.points
                .filter(
                  (p) =>
                    Date.parse(p.time) +
                      (traffic.resolution === "minute" ? 60000 : 3600000) >
                    start,
                )
                .map((p) => ({ ...p, hour: p.time }));
              return (
                <Card key={channel} className="min-w-0 p-5">
                  <h3 className="flex items-center gap-2 font-medium">
                    <Activity className="size-4" />
                    {channel === "api"
                      ? "API requests"
                      : "Public page requests"}
                  </h3>
                  <div className="my-4 grid grid-cols-3 gap-3 text-sm">
                    <div>
                      <p className="text-xs text-muted-foreground">Requests</p>
                      <p className="mt-1 text-xl tabular-nums">
                        {data.summary.requests.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">4xx / 5xx</p>
                      <p className="mt-1 tabular-nums">
                        {data.summary.errors_4xx} / {data.summary.errors_5xx}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">
                        Mean / p95
                      </p>
                      <p className="mt-1 tabular-nums">
                        {data.summary.requests
                          ? `${data.summary.average_ms} ms`
                          : "—"}{" "}
                        /{" "}
                        {data.summary.p95_upper_ms == null
                          ? "—"
                          : data.summary.p95_upper_ms < 0
                            ? "> 60 s"
                            : `≤ ${data.summary.p95_upper_ms} ms`}
                      </p>
                    </div>
                  </div>
                  <TrafficChart
                    rows={rows}
                    label={`${channel === "api" ? "API" : "Public website"} requests and errors per ${traffic.resolution}`}
                  />
                  <details className="mt-3 text-xs">
                    <summary className="cursor-pointer text-muted-foreground">
                      Show values
                    </summary>
                    <div className="mt-2 max-h-48 overflow-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>UTC</TableHead>
                            <TableHead>Requests</TableHead>
                            <TableHead>4xx</TableHead>
                            <TableHead>5xx</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {rows.map((row) => (
                            <TableRow key={row.hour}>
                              <TableCell>{date(row.hour)}</TableCell>
                              <TableCell>{row.requests}</TableCell>
                              <TableCell>{row.errors_4xx}</TableCell>
                              <TableCell>{row.errors_5xx}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </details>
                </Card>
              );
            })}
          </div>
        )}
        {user?.permissions.includes("api_stats.read") && (
          <Button variant="outline" asChild>
            <Link href="/dashboard/api-stats">Detailed API statistics</Link>
          </Button>
        )}
      </section>
      {project && (
        <>
          <section className="space-y-3">
            <h2 className="text-lg font-medium">Clio · Observation sources</h2>
            {project.observations?.error && (
              <Message>{project.observations.error}</Message>
            )}
            <Card className="overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Source</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Latest observation</TableHead>
                    <TableHead>Response received</TableHead>
                    <TableHead>Successfully saved</TableHead>
                    <TableHead>Last attempt</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.values(project.observations?.sources ?? {}).map(
                    (source) => (
                      <TableRow key={source.source_id}>
                        <TableCell>
                          <p className="font-medium">{source.label}</p>
                          {source.consecutive_failures > 0 && (
                            <p className="mt-1 max-w-64 text-xs text-amber-200">
                              {source.last_error_message}
                            </p>
                          )}
                        </TableCell>
                        <TableCell>
                          <Status value={old ? "unknown" : source.status} />
                        </TableCell>
                        <TableCell>
                          <Stamp value={source.latest_observation_at} />
                          {source.latest_interval_end && (
                            <p className="text-xs text-muted-foreground">
                              to {date(source.latest_interval_end)}
                            </p>
                          )}
                        </TableCell>
                        <TableCell>
                          <Stamp value={source.last_response_at} />
                        </TableCell>
                        <TableCell>
                          <Stamp value={source.last_success_at} />
                        </TableCell>
                        <TableCell>
                          <Stamp value={source.last_attempt_at} />
                        </TableCell>
                      </TableRow>
                    ),
                  )}
                </TableBody>
              </Table>
            </Card>
            <p className="text-xs text-muted-foreground">
              A successful response can contain old observations. Receipt and
              observation times are shown separately.
            </p>
          </section>
          <section className="space-y-3">
            <h2 className="text-lg font-medium">
              Clio · Model input observations
            </h2>
            <p className="text-xs text-muted-foreground">
              Last scheduled refresh completed:{" "}
              <Stamp value={project.observations?.last_refresh_completed_at} />.
              Receipt times are recorded from the monitoring rollout onward.
            </p>
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
            <h2 className="text-lg font-medium">Prophet · Forecast products</h2>
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
