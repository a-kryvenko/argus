import type { Metadata } from "next";
import { cookies } from "next/headers";
import DashboardShell from "./DashboardShell";
export const metadata: Metadata = {
  title: "Dashboard",
  robots: { index: false, follow: false },
};
export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const cookieStore = await cookies();
  return (
    <DashboardShell
      defaultOpen={cookieStore.get("sidebar_state")?.value !== "false"}
    >
      {children}
    </DashboardShell>
  );
}
