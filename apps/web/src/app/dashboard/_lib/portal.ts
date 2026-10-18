// Keep Radix portals inside the dashboard's scoped theme and utility selectors.
export function dashboardPortal() {
  return typeof document === "undefined"
    ? undefined
    : document.getElementById("dashboard-portals");
}
