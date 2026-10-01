"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft, ArrowRight, Loader2, LockKeyhole } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Message } from "../_components/presentation";
import { dashboardRequest } from "../session";

export default function Login() {
  const router = useRouter();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <main id="dashboard-content" className="dashboard-login">
      <div className="login-card">
        <div className="mb-6 flex items-center gap-3 text-xs uppercase tracking-widest text-muted-foreground"><LockKeyhole className="size-4 text-primary" />Account access</div>
        <h1 className="text-2xl font-semibold tracking-tight">Welcome back</h1>
        <p className="mb-6 mt-2 text-sm text-muted-foreground">Sign in to your Argus workspace.</p>
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
        <p className="mt-6 text-xs leading-relaxed text-muted-foreground">Need an account or a password reset? Contact your administrator.</p>
        <Link href="/" className="mt-6 flex items-center gap-2 text-xs text-primary"><ArrowLeft className="size-3.5" />Forecast overview</Link>
      </div>
    </main>
  );
}
