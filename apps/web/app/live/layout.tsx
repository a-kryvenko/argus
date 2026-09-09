import type { ReactNode } from "react";
export const metadata = {
  title: "Live observations",
  description: "Current solar wind observations, Kp and Dst indices in UTC.",
};
export default function LiveLayout({ children }: { children: ReactNode }) {
  return children;
}
