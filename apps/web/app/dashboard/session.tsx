"use client";
import { createContext, useContext } from "react";
import { apiRequest } from "../_utils/api";

export type User = {
  id: number;
  username: string;
  active: boolean;
  groups: string[];
  permissions: string[];
};
export const SessionContext = createContext<User | null>(null);
export const useSession = () => useContext(SessionContext);
export function dashboardRequest<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  return apiRequest<T>(`/dashboard${path}`, {
    method,
    cache: "no-store",
    credentials: "same-origin",
    ...(body === undefined
      ? {}
      : {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
}
