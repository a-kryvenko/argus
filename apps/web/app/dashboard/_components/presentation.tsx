import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Inbox,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export function PageHeading({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="space-y-1.5">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">{description}</p>
      </div>
      {action}
    </div>
  );
}
export function Message({
  children,
  success = false,
}: {
  children: React.ReactNode;
  success?: boolean;
}) {
  return (
    <div
      role={success ? "status" : "alert"}
      className={cn(
        "flex items-start gap-3 rounded-lg border px-4 py-3 text-sm",
        success
          ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-300"
          : "border-red-400/20 bg-red-400/5 text-red-300",
      )}
    >
      <AlertCircle className="mt-0.5 size-4" />
      <div>{children}</div>
    </div>
  );
}
export function EmptyState({
  title = "No results",
  description,
}: {
  title?: string;
  description: string;
}) {
  return (
    <div className="flex min-h-52 flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      <div className="rounded-full border bg-muted/50 p-3">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div>
        <p className="font-medium">{title}</p>
        <p className="mt-1 max-w-sm text-sm text-muted-foreground">
          {description}
        </p>
      </div>
    </div>
  );
}
export function TableLoading() {
  return (
    <div role="status" aria-label="Loading data" className="space-y-5 p-6">
      {Array.from({ length: 6 }, (_, i) => (
        <Skeleton key={i} className="h-5 w-full" />
      ))}
    </div>
  );
}
export function Pending({ label = "Loading…" }: { label?: string }) {
  return (
    <span
      role="status"
      className="flex items-center gap-2 text-sm text-muted-foreground"
    >
      <Loader2 className="size-4 animate-spin" />
      {label}
    </span>
  );
}
export function Pagination({
  page,
  size,
  total,
  busy = false,
  onChange,
}: {
  page: number;
  size: number;
  total: number;
  busy?: boolean;
  onChange: (page: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / size));
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-t px-5 py-4">
      <span className="text-xs text-muted-foreground">
        {total
          ? `${((page - 1) * size + 1).toLocaleString()}–${Math.min(page * size, total).toLocaleString()} of ${total.toLocaleString()}`
          : "0 results"}
      </span>
      <div className="flex items-center gap-3">
        <span className="text-xs text-muted-foreground">
          Page {page} of {pages}
        </span>
        <Button
          variant="outline"
          size="icon"
          className="size-8"
          aria-label="Previous page"
          disabled={busy || page <= 1}
          onClick={() => onChange(page - 1)}
        >
          <ChevronLeft className="size-4" />
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="size-8"
          aria-label="Next page"
          disabled={busy || page >= pages}
          onClick={() => onChange(page + 1)}
        >
          <ChevronRight className="size-4" />
        </Button>
      </div>
    </div>
  );
}
export function MetricCard({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: React.ReactNode;
  detail: string;
  icon: React.ReactNode;
}) {
  return (
    <Card className="shadow-none">
      <CardContent className="p-5">
        <div className="flex items-center justify-between gap-2 text-muted-foreground">
          <span className="text-xs font-medium">{label}</span>
          {icon}
        </div>
        <div className="mt-3 text-3xl font-semibold tracking-tight tabular-nums">
          {value}
        </div>
        <p className="mt-2 text-xs text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  );
}
export const selectClass =
  "flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50";
