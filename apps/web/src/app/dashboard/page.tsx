"use client";
import { useSession } from "./session";
import ProjectOverview from "./_components/ProjectOverview";
import ClientOverview from "./_components/ClientOverview";

export default function Dashboard() {
  const user = useSession();
  if (!user) return null;
  return user.permissions.includes("project_monitoring.read") ? (
    <ProjectOverview />
  ) : (
    <ClientOverview />
  );
}
