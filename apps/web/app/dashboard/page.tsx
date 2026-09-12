"use client";
import Link from "next/link";
import {
  ArrowUpRight,
  Database,
  Layers3,
  ShieldCheck,
  UsersRound,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeading } from "./_components/presentation";
import ApiActivity from "./_components/ApiActivity";
import { useSession } from "./session";

const resources = [
  {
    title: "Original observations",
    description: "Browse measurements by metric and time.",
    href: "/dashboard/observations",
    permission: "observations.read",
    icon: Database,
  },
  {
    title: "Normalized data",
    description: "Explore hourly inputs for forecasting.",
    href: "/dashboard/observations/normalized",
    permission: "observations.read",
    icon: Layers3,
  },
  {
    title: "Users & access",
    description: "Manage accounts and group membership.",
    href: "/dashboard/users",
    permission: "users.manage",
    icon: UsersRound,
  },
];
export default function Dashboard() {
  const user = useSession();
  const available = resources.filter((resource) =>
    user?.permissions.includes(resource.permission),
  );
  return (
    <>
      <PageHeading
        title="Workspace overview"
        description={`Welcome back, ${user?.username ?? ""}. Here’s what’s happening across your workspace.`}
        action={
          <Badge
            variant="outline"
            className="gap-1.5 px-3 py-1.5 font-normal text-muted-foreground"
          >
            <ShieldCheck className="size-3.5 text-primary" />
            {user?.groups.join(", ") || "Member"}
          </Badge>
        }
      />
      {user?.permissions.includes("api_stats.read") && <ApiActivity compact />}
      {available.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-sm font-medium">Explore your workspace</h2>
          <div className="grid gap-4 md:grid-cols-3">
            {available.map((resource) => (
              <Link
                key={resource.href}
                href={resource.href}
                className="group rounded-xl focus-visible:outline-primary"
              >
                <Card className="h-full shadow-none transition-colors group-hover:border-primary/40">
                  <CardContent className="p-5">
                    <div className="mb-5 flex items-center justify-between">
                      <div className="flex size-9 items-center justify-center rounded-lg border bg-muted/50">
                        <resource.icon className="size-4 text-primary" />
                      </div>
                      <ArrowUpRight className="size-4 text-muted-foreground transition-colors group-hover:text-primary" />
                    </div>
                    <h3 className="text-sm font-medium">{resource.title}</h3>
                    <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
                      {resource.description}
                    </p>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}
      {!user?.permissions.length && (
        <Card className="p-6">
          <h2 className="font-medium">Your account is ready</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Contact an administrator to assign access to workspace data.
          </p>
        </Card>
      )}
    </>
  );
}
