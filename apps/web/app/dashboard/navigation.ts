import {
  Activity,
  Database,
  Layers3,
  LayoutDashboard,
  UsersRound,
} from "lucide-react";
export const navigation = [
  {
    title: "Analytics",
    items: [
      {
        href: "/dashboard",
        title: "Overview",
        icon: LayoutDashboard,
        permission: null,
      },
      {
        href: "/dashboard/api-stats",
        title: "API statistics",
        icon: Activity,
        permission: "api_stats.read",
      },
    ],
  },
  {
    title: "Observations",
    items: [
      {
        href: "/dashboard/observations",
        title: "Original data",
        icon: Database,
        permission: "observations.read",
      },
      {
        href: "/dashboard/observations/normalized",
        title: "Normalized data",
        icon: Layers3,
        permission: "observations.read",
      },
    ],
  },
  {
    title: "Administration",
    items: [
      {
        href: "/dashboard/users",
        title: "Users & access",
        icon: UsersRound,
        permission: "users.manage",
      },
    ],
  },
];
export const dashboardLinks = navigation.flatMap((group) => group.items);
