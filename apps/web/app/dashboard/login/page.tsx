"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft, ArrowRight, Loader2, LockKeyhole, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Message } from "../_components/presentation";
import { dashboardRequest } from "../session";

export default function Login() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <main className="grid min-h-svh lg:grid-cols-2">
      <div className="relative hidden flex-col justify-between overflow-hidden border-r bg-sidebar p-12 lg:flex">
        <Link
          href="/"
          className="flex items-center gap-3 text-sm font-semibold tracking-widest"
        >
          <Sun className="size-6 text-primary" />
          ARGUS SUNWATCH
        </Link>
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 flex items-center justify-center"
        >
          <div className="relative flex size-[420px] items-center justify-center rounded-full border border-primary/10">
            <div className="absolute size-[300px] rounded-full border border-primary/15" />
            <div className="absolute size-[180px] rounded-full border border-primary/20 bg-primary/5" />
            <Sun className="size-20 text-primary/60" />
            <div className="absolute right-12 top-16 size-3 rounded-full bg-primary shadow-[0_0_30px_hsl(var(--primary)/.6)]" />
          </div>
        </div>
        <div className="relative max-w-md space-y-4">
          <span className="text-xs uppercase tracking-[.2em] text-primary">
            Your observation workspace
          </span>
          <h1 className="text-4xl font-medium leading-tight tracking-tight">
            A clearer view of
            <br />
            space weather.
          </h1>
          <p className="text-sm leading-relaxed text-muted-foreground">
            Explore observations, monitor API activity and manage access in one
            place.
          </p>
        </div>
      </div>
      <div className="flex flex-col p-6 sm:p-10">
        <Link
          href="/"
          className="flex w-fit items-center gap-2 text-xs text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-3.5" />
          Back to public site
        </Link>
        <div className="flex flex-1 items-center justify-center py-12">
          <Card className="w-full max-w-sm border-0 bg-transparent shadow-none">
            <CardHeader className="space-y-3 px-0">
              <div className="mb-3 flex size-11 items-center justify-center rounded-xl border bg-muted/50">
                <LockKeyhole className="size-5 text-primary" />
              </div>
              <CardTitle className="text-2xl font-semibold tracking-tight">
                Welcome back
              </CardTitle>
              <CardDescription>
                Sign in to your Argus workspace.
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0 pt-4">
              <form
                className="space-y-5"
                onSubmit={async (e) => {
                  e.preventDefault();
                  setBusy(true);
                  setError("");
                  const data = new FormData(e.currentTarget);
                  try {
                    await dashboardRequest("/login", "POST", {
                      username: data.get("username"),
                      password: data.get("password"),
                    });
                    router.replace("/dashboard");
                  } catch (e) {
                    setError(e instanceof Error ? e.message : "Sign-in failed");
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <Input
                    id="username"
                    name="username"
                    placeholder="Your username"
                    autoComplete="username"
                    required
                    maxLength={80}
                    className="h-10"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    name="password"
                    type="password"
                    placeholder="Enter your password"
                    autoComplete="current-password"
                    required
                    maxLength={256}
                    className="h-10"
                  />
                </div>
                {error && <Message>{error}</Message>}
                <Button className="h-10 w-full" disabled={busy}>
                  {busy ? (
                    <Loader2 className="mr-2 size-4 animate-spin" />
                  ) : null}
                  {busy ? "Signing in…" : "Sign in"}
                  {!busy && <ArrowRight className="ml-2 size-4" />}
                </Button>
              </form>
              <p className="mt-6 text-center text-xs leading-relaxed text-muted-foreground">
                Need an account or a password reset?
                <br />
                Contact your administrator.
              </p>
            </CardContent>
          </Card>
        </div>
        <p className="text-center text-[11px] text-muted-foreground">
          ARGUS Sunwatch · Observation & analytics workspace
        </p>
      </div>
    </main>
  );
}
