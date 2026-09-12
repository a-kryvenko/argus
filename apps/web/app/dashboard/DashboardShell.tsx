"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Globe, ShieldCheck, Sun } from "lucide-react";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import {
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbSeparator,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import { ApiError } from "../_utils/api";
import { dashboardRequest, SessionContext, type User } from "./session";
import { AppSidebar } from "./_components/AppSidebar";
import { Message, Pending } from "./_components/presentation";
import { dashboardLinks } from "./navigation";
import "./dashboard.css";

export default function DashboardShell({
  children,
  defaultOpen = true,
}: {
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const pathname = usePathname().replace(/\/$/, "");
  const router = useRouter();
  const login = pathname === "/dashboard/login";
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState("");
  const [signingOut, setSigningOut] = useState(false);
  useEffect(() => {
    if (login) return;
    let disposed = false;
    const refresh = () =>
      dashboardRequest<User>("/me")
        .then((value) => {
          if (!disposed) {
            setUser(value);
            setError("");
          }
        })
        .catch((e: unknown) => {
          if (disposed) return;
          setUser(null);
          if (e instanceof ApiError && e.status === 401)
            router.replace("/dashboard/login");
          else
            setError(e instanceof Error ? e.message : "Could not load session");
        });
    void refresh();
    const timer = window.setInterval(refresh, 60000);
    window.addEventListener("focus", refresh);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      window.removeEventListener("focus", refresh);
    };
  }, [login, pathname, router]);
  async function logout() {
    setSigningOut(true);
    try {
      await dashboardRequest("/logout", "POST");
      setUser(null);
      router.replace("/dashboard/login");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not sign out");
    } finally {
      setSigningOut(false);
    }
  }
  const current = dashboardLinks.find((link) => link.href === pathname);
  return (
    <div className="dashboard dark">
      <div id="dashboard-portals" />
      {login ? (
        children
      ) : !user ? (
        <div className="flex min-h-svh flex-col items-center justify-center gap-6 px-6">
          <Sun className="size-8 text-primary" />
          {error ? (
            <>
              <Message>{error}</Message>
              <Button asChild variant="outline">
                <Link href="/dashboard/login">Return to sign in</Link>
              </Button>
            </>
          ) : (
            <Pending label="Opening your workspace…" />
          )}
        </div>
      ) : (
        <SessionContext value={user}>
          <SidebarProvider
            defaultOpen={defaultOpen}
            style={{ "--sidebar-width": "15rem" } as React.CSSProperties}
          >
            <AppSidebar user={user} logout={logout} signingOut={signingOut} />
            <SidebarInset className="min-w-0 overflow-hidden border border-border/70 shadow-none">
              <header className="flex h-16 shrink-0 items-center justify-between gap-3 border-b px-4 sm:px-6">
                <div className="flex min-w-0 items-center gap-3">
                  <SidebarTrigger className="-ml-1" />
                  <Separator orientation="vertical" className="h-4" />
                  <Breadcrumb>
                    <BreadcrumbList>
                      <BreadcrumbItem className="hidden sm:block">
                        <BreadcrumbLink asChild>
                          <Link href="/dashboard">Workspace</Link>
                        </BreadcrumbLink>
                      </BreadcrumbItem>
                      <BreadcrumbSeparator className="hidden sm:block" />
                      <BreadcrumbItem>
                        <BreadcrumbPage>
                          {current?.title ?? "Dashboard"}
                        </BreadcrumbPage>
                      </BreadcrumbItem>
                    </BreadcrumbList>
                  </Breadcrumb>
                </div>
                <div className="flex items-center gap-4">
                  <span className="hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex">
                    <Globe className="size-3.5" />
                    UTC
                  </span>
                  <span className="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs text-muted-foreground">
                    <ShieldCheck className="size-3.5 text-primary" />
                    {user.groups.includes("admins")
                      ? "Administrator"
                      : "Member"}
                  </span>
                </div>
              </header>
              <div className="mx-auto flex w-full max-w-[1680px] flex-1 flex-col gap-6 p-4 sm:p-6 lg:p-8">
                {error && <Message>{error}</Message>}
                {current?.permission &&
                !user.permissions.includes(current.permission) ? (
                  <Message>You do not have access to this section.</Message>
                ) : (
                  children
                )}
              </div>
              <footer className="flex flex-wrap items-center justify-between gap-2 border-t px-6 py-4 text-[11px] text-muted-foreground">
                <span>ARGUS Sunwatch</span>
                <span>Space weather · Observations & analytics</span>
              </footer>
            </SidebarInset>
          </SidebarProvider>
        </SessionContext>
      )}
    </div>
  );
}
