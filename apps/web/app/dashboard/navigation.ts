import {
  Activity,
  LayoutDashboard,
  UsersRound,
  Satellite,
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
    title: "Risk assessments",
    items: [
      {
        href: "/dashboard/risk/leo",
        title: "LEO drag",
        icon: Satellite,
        permission: null,
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
