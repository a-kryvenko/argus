"use client";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Clock3,
  Gauge,
  RotateCcw,
  ServerCrash,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { dashboardRequest } from "../session";
import { EmptyState, Message, MetricCard, selectClass } from "./presentation";
const TrafficChart = dynamic(() => import("./TrafficChart"), {
  ssr: false,
  loading: () => <Skeleton className="h-64 w-full" />,
});
export type Summary = {
  requests: number;
  errors_4xx: number;
  errors_5xx: number;
  average_ms: number;
  p95_upper_ms: number | null;
};
export type Stats = {
  summary: Summary;
  routes: (Summary & { route: string; method: string })[];
  hours: (Summary & { hour: string })[];
  statuses: { status: number; requests: number }[];
};
const p95 = (value: number | null) =>
  value === null
    ? "—"
    : value === -1
      ? "> 60 s"
      : `≤ ${value.toLocaleString()} ms`;

export default function ApiActivity({
  compact = false,
}: {
  compact?: boolean;
}) {
  const [hours, setHours] = useState(24);
  const [version, setVersion] = useState(0);
  const [data, setData] = useState<Stats | null>(null);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  useEffect(() => {
    let disposed = false;
    const refresh = () =>
      dashboardRequest<Stats>(`/api-stats?hours=${hours}`)
        .then((v) => {
          if (!disposed) {
            setData(v);
            setError("");
            setUpdatedAt(new Date());
          }
        })
        .catch((e) => {
          if (!disposed) setError(e.message);
        });
    void refresh();
    const timer = window.setInterval(refresh, 30000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [hours, version]);
  const controls = (
    <div className="flex items-center gap-2">
      <select
        aria-label="Statistics period"
        value={hours}
        className={`${selectClass} w-32`}
        onChange={(e) => {
          setData(null);
          setHours(Number(e.target.value));
        }}
      >
        <option value={24}>24 hours</option>
        <option value={168}>7 days</option>
        <option value={720}>30 days</option>
      </select>
      <Button
        variant="outline"
        size="icon"
        aria-label="Refresh statistics"
        onClick={() => setVersion((v) => v + 1)}
      >
        <RotateCcw className="size-3.5" />
      </Button>
    </div>
  );
  return (
    <>
      {error && (
        <Message>
          {error}
          {data && " Showing the last successful update."}
        </Message>
      )}
      {!data ? (
        <div
          role="status"
          aria-label="Loading statistics"
          className="space-y-4"
        >
          <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-36 rounded-xl" />
            ))}
          </div>
          <Skeleton className="h-80 rounded-xl" />
          {error && (
            <Button variant="outline" onClick={() => setVersion((v) => v + 1)}>
              Retry
            </Button>
          )}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="Total requests"
              value={data.summary.requests.toLocaleString()}
              detail={`Across all API routes · ${hours === 24 ? "24 hours" : `${hours / 24} days`}`}
              icon={<ArrowUpRight className="size-4" />}
            />
            <MetricCard
              label="Average response"
              value={
                <>
                  {data.summary.average_ms.toLocaleString()}
                  <span className="ml-1 text-base font-normal text-muted-foreground">
                    ms
                  </span>
                </>
              }
              detail="Application processing time"
              icon={<Clock3 className="size-4" />}
            />
            <MetricCard
              label="Error responses"
              value={(
                data.summary.errors_4xx + data.summary.errors_5xx
              ).toLocaleString()}
              detail={`${data.summary.errors_4xx.toLocaleString()} client (4xx) · ${data.summary.errors_5xx.toLocaleString()} server (5xx)`}
              icon={<ServerCrash className="size-4" />}
            />
            <MetricCard
              label="p95 response time"
              value={
                <span className="text-2xl">
                  {p95(data.summary.p95_upper_ms)}
                </span>
              }
              detail="Estimated upper bound"
              icon={<Gauge className="size-4" />}
            />
          </div>
          <Card className="overflow-hidden shadow-none">
            <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-4 space-y-0 border-b px-5 py-5">
              <div className="space-y-1.5">
                <CardTitle className="text-sm">API activity</CardTitle>
                <CardDescription className="text-xs">
                  Request volume and errors by hour
                </CardDescription>
              </div>
              {controls}
            </CardHeader>
            <CardContent className="px-3 pb-1 pt-4 sm:px-5">
              {data.summary.requests ? (
                <TrafficChart rows={data.hours} />
              ) : (
                <EmptyState
                  title="Waiting for API activity"
                  description="Request statistics will appear here as traffic reaches the API."
                />
              )}
            </CardContent>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t px-5 py-3 text-[11px] text-muted-foreground">
              <span className="flex items-center gap-2">
                <Activity className="size-3.5" />
                {error ? "Update paused" : "Refreshes every 30 seconds"}
              </span>
              <span>
                {updatedAt
                  ? `Updated ${updatedAt.toISOString().slice(11, 19)} UTC`
                  : "All times UTC"}
              </span>
            </div>
          </Card>
          {!compact && (
            <>
              <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_280px]">
                <Card className="min-w-0 overflow-hidden shadow-none">
                  <CardHeader className="border-b px-5 py-5">
                    <CardTitle className="text-sm">
                      Endpoint performance
                    </CardTitle>
                    <CardDescription className="text-xs">
                      Traffic grouped by route and method
                    </CardDescription>
                  </CardHeader>
                  {data.routes.length ? (
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-muted/40">
                          <TableHead className="px-5">Endpoint</TableHead>
                          <TableHead className="text-right">Requests</TableHead>
                          <TableHead className="text-right">
                            4xx / 5xx
                          </TableHead>
                          <TableHead className="text-right">Average</TableHead>
                          <TableHead className="px-5 text-right">
                            p95 bound
                          </TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {[...data.routes]
                          .sort((a, b) => b.requests - a.requests)
                          .map((row) => (
                            <TableRow key={`${row.method}:${row.route}`}>
                              <TableCell className="whitespace-nowrap px-5 py-3.5">
                                <Badge
                                  variant="outline"
                                  className="mr-2 font-mono text-[10px] font-normal text-primary"
                                >
                                  {row.method}
                                </Badge>
                                <span className="font-mono text-xs">
                                  {row.route}
                                </span>
                              </TableCell>
                              <TableCell className="text-right text-xs tabular-nums">
                                {row.requests.toLocaleString()}
                              </TableCell>
                              <TableCell className="whitespace-nowrap text-right text-xs tabular-nums text-muted-foreground">
                                {row.errors_4xx} / {row.errors_5xx}
                              </TableCell>
                              <TableCell className="whitespace-nowrap text-right text-xs tabular-nums">
                                {row.average_ms} ms
                              </TableCell>
                              <TableCell className="whitespace-nowrap px-5 text-right text-xs tabular-nums">
                                {p95(row.p95_upper_ms)}
                              </TableCell>
                            </TableRow>
                          ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <EmptyState description="No endpoints have recorded activity in this period." />
                  )}
                </Card>
                <Card className="shadow-none">
                  <CardHeader className="px-5 py-5">
                    <CardTitle className="text-sm">Response codes</CardTitle>
                    <CardDescription className="text-xs">
                      Distribution by HTTP status
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5 px-5">
                    {data.statuses.length ? (
                      data.statuses.map((status) => (
                        <div key={status.status} className="space-y-2">
                          <div className="flex justify-between text-xs">
                            <span className="font-mono">{status.status}</span>
                            <span className="tabular-nums text-muted-foreground">
                              {status.requests.toLocaleString()}{" "}
                              <span className="ml-2">
                                {(
                                  (status.requests / data.summary.requests) *
                                  100
                                ).toFixed(1)}
                                %
                              </span>
                            </span>
                          </div>
                          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                            <div
                              className={`h-full rounded-full ${status.status >= 500 ? "bg-rose-400" : status.status >= 400 ? "bg-amber-400" : "bg-primary"}`}
                              style={{
                                width: `${(status.requests / data.summary.requests) * 100}%`,
                              }}
                            />
                          </div>
                        </div>
                      ))
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        No responses recorded.
                      </p>
                    )}
                  </CardContent>
                </Card>
              </div>
              <Card className="overflow-hidden shadow-none">
                <details>
                  <summary className="cursor-pointer px-5 py-4 text-sm font-medium">
                    Hourly activity{" "}
                    <span className="ml-2 text-xs font-normal text-muted-foreground">
                      View detailed values
                    </span>
                  </summary>
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/40">
                        <TableHead className="px-5">Hour · UTC</TableHead>
                        <TableHead>Requests</TableHead>
                        <TableHead>4xx</TableHead>
                        <TableHead>5xx</TableHead>
                        <TableHead>Average</TableHead>
                        <TableHead>p95 bound</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {[...data.hours].reverse().map((row) => (
                        <TableRow key={row.hour}>
                          <TableCell className="px-5 font-mono text-xs">
                            {row.hour.replace("T", " ").slice(0, 16)}
                          </TableCell>
                          <TableCell>{row.requests.toLocaleString()}</TableCell>
                          <TableCell>{row.errors_4xx}</TableCell>
                          <TableCell>{row.errors_5xx}</TableCell>
                          <TableCell>{row.average_ms} ms</TableCell>
                          <TableCell>{p95(row.p95_upper_ms)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </details>
              </Card>
              <p className="max-w-3xl text-xs leading-relaxed text-muted-foreground">
                All API routes are included, with up to 15 seconds of reporting
                delay. Data is retained for 30 days. p95 is estimated from
                latency buckets; response times exclude network transfer.
              </p>
            </>
          )}
        </>
      )}
    </>
  );
}
