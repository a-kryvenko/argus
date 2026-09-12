"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import {
  ArrowDownWideNarrow,
  CalendarRange,
  Database,
  Layers3,
  RotateCcw,
  SlidersHorizontal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  EmptyState,
  Message,
  PageHeading,
  Pagination,
  selectClass,
  TableLoading,
} from "../_components/presentation";
import { dashboardRequest } from "../session";

type Result = {
  columns: string[];
  items: Record<string, string | number | null>[];
  total: number;
  page: number;
  page_size: number;
};
const headings: Record<string, string> = {
  id: "ID",
  observed_at: "Observed at · UTC",
  metric: "Metric",
  value: "Value",
  bx: "Bx",
  by: "By",
  bz: "Bz",
  v: "Speed",
  n: "Density",
  t: "Temperature",
  kp: "Kp",
  dst: "Dst",
  ap: "Ap",
  f10_7: "F10.7",
  s10: "S10",
  m10: "M10",
  y10: "Y10",
};
function format(value: string | number | null, column: string) {
  if (value === null) return "—";
  if (column === "observed_at")
    return new Date(value).toISOString().replace("T", " ").replace(".000Z", "");
  return String(value);
}
export default function ObservationTable({
  normalized = false,
}: {
  normalized?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [version, setVersion] = useState(0);
  const [data, setData] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  useEffect(() => {
    let disposed = false;
    dashboardRequest<Result>(
      `/observations?kind=${normalized ? "normalized" : "raw"}&page=${page}&${query}`,
    )
      .then((v) => {
        if (!disposed) {
          setData(v);
          setError("");
        }
      })
      .catch((e) => {
        if (!disposed) {
          setData(null);
          setError(e.message);
        }
      })
      .finally(() => {
        if (!disposed) setBusy(false);
      });
    return () => {
      disposed = true;
    };
  }, [normalized, page, query, version]);
  function changePage(value: number) {
    setBusy(true);
    setPage(value);
  }
  return (
    <>
      <PageHeading
        title="Observations"
        description="Explore stored measurements and the normalized inputs used for forecasting."
        action={
          <Button
            variant="outline"
            disabled={busy}
            onClick={() => {
              setBusy(true);
              setVersion((v) => v + 1);
            }}
          >
            <RotateCcw className="mr-2 size-3.5" />
            Refresh
          </Button>
        }
      />
      <div className="flex w-fit gap-1 rounded-lg border bg-muted/40 p-1">
        <Button variant={!normalized ? "secondary" : "ghost"} size="sm" asChild>
          <Link
            href="/dashboard/observations"
            aria-current={!normalized ? "page" : undefined}
          >
            <Database className="mr-2 size-3.5" />
            Original data
          </Link>
        </Button>
        <Button variant={normalized ? "secondary" : "ghost"} size="sm" asChild>
          <Link
            href="/dashboard/observations/normalized"
            aria-current={normalized ? "page" : undefined}
          >
            <Layers3 className="mr-2 size-3.5" />
            Normalized
          </Link>
        </Button>
      </div>
      <Card className="shadow-none">
        <CardContent className="p-5">
          <div className="mb-4 flex items-center gap-2 text-xs font-medium">
            <SlidersHorizontal className="size-3.5 text-muted-foreground" />
            Filter observations
          </div>
          <form
            className="flex flex-wrap items-end gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              const form = new FormData(e.currentTarget);
              const params = new URLSearchParams();
              for (const key of ["start", "end"]) {
                const value = String(form.get(key) || "");
                if (value) params.set(key, new Date(`${value}Z`).toISOString());
              }
              for (const key of ["order", "page_size", "metric"]) {
                const value = String(form.get(key) || "").trim();
                if (value) params.set(key, value);
              }
              if (
                params.get("start") &&
                params.get("end") &&
                params.get("start")! > params.get("end")!
              ) {
                setError("Start must precede end");
                return;
              }
              setError("");
              setBusy(true);
              setPage(1);
              setQuery(params.toString());
              setVersion((v) => v + 1);
            }}
          >
            <div className="min-w-40 flex-1 space-y-2">
              <Label htmlFor="start" className="text-xs text-muted-foreground">
                From (UTC)
              </Label>
              <Input id="start" type="datetime-local" name="start" />
            </div>
            <div className="min-w-40 flex-1 space-y-2">
              <Label htmlFor="end" className="text-xs text-muted-foreground">
                To (UTC)
              </Label>
              <Input id="end" type="datetime-local" name="end" />
            </div>
            {!normalized && (
              <div className="w-28 space-y-2">
                <Label
                  htmlFor="metric"
                  className="text-xs text-muted-foreground"
                >
                  Metric
                </Label>
                <Input
                  id="metric"
                  name="metric"
                  placeholder="All metrics"
                  maxLength={16}
                />
              </div>
            )}
            <div className="w-36 space-y-2">
              <Label htmlFor="order" className="text-xs text-muted-foreground">
                Time order
              </Label>
              <select id="order" name="order" className={selectClass}>
                <option value="desc">Newest first</option>
                <option value="asc">Oldest first</option>
              </select>
            </div>
            <div className="w-20 space-y-2">
              <Label
                htmlFor="page_size"
                className="text-xs text-muted-foreground"
              >
                Rows
              </Label>
              <select id="page_size" name="page_size" className={selectClass}>
                <option>50</option>
                <option>100</option>
                <option>200</option>
              </select>
            </div>
            <Button disabled={busy}>Apply filters</Button>
            <Button
              type="button"
              variant="ghost"
              disabled={busy}
              onClick={(event) => {
                event.currentTarget.form?.reset();
                setBusy(true);
                setPage(1);
                setQuery("");
                setVersion((v) => v + 1);
              }}
            >
              Reset
            </Button>
          </form>
        </CardContent>
      </Card>
      {error && <Message>{error}</Message>}
      <Card className="overflow-hidden shadow-none">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b px-5 py-4">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold">
              {normalized ? "Normalized observations" : "Original measurements"}
            </h2>
            {data && (
              <Badge variant="secondary" className="font-normal tabular-nums">
                {data.total.toLocaleString()} records
              </Badge>
            )}
          </div>
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <CalendarRange className="size-3.5" />
            UTC timestamps
          </span>
        </div>
        {busy ? (
          <TableLoading />
        ) : data?.items.length ? (
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/40 hover:bg-muted/40">
                {data.columns.map((c) => (
                  <TableHead
                    key={c}
                    className="h-11 whitespace-nowrap px-5 text-xs"
                  >
                    {headings[c] ?? c}
                    {c === "observed_at" && (
                      <ArrowDownWideNarrow className="ml-2 inline-block size-3" />
                    )}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.items.map((row, i) => (
                <TableRow key={String(row.id ?? row.observed_at ?? i)}>
                  {data.columns.map((c) => (
                    <TableCell
                      key={c}
                      className={`whitespace-nowrap px-5 py-3.5 font-mono text-xs tabular-nums ${c === "id" || row[c] === null ? "text-muted-foreground" : ""}`}
                    >
                      {c === "metric" ? (
                        <Badge
                          variant="outline"
                          className="font-mono font-normal"
                        >
                          {String(row[c])}
                        </Badge>
                      ) : (
                        format(row[c], c)
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <EmptyState
            title={error ? "Data unavailable" : "No observations found"}
            description={
              error
                ? "Try refreshing the data."
                : "Try another date range or clear the metric filter."
            }
          />
        )}
        {data && (
          <Pagination
            page={page}
            size={data.page_size}
            total={data.total}
            busy={busy}
            onChange={changePage}
          />
        )}
      </Card>
      <p className="text-xs text-muted-foreground">
        {normalized
          ? "Hourly normalized inputs for forecast generation. Missing values are shown as a dash."
          : "Stored measurements by metric. Values retain their original precision."}
      </p>
    </>
  );
}
