import DragRiskIcon from '@/components/DragRiskIcon';
import {
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
    ],
  },
  {
    title: "Risk assessments",
    items: [
      {
        href: "/dashboard/risk/leo",
        title: "Drag risk",
        icon: DragRiskIcon,
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
export const dashboardLinks = navigation.flatMap<(typeof navigation)[number]["items"][number]>((group) => group.items);
