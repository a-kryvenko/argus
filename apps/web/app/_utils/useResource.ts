"use client";
import { useEffect, useState } from "react";
import { apiRequest } from "./api";

export function useResource<T>(path: string) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<{
    path: string;
    attempt: number;
    data: T | null;
    error: string | null;
  } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const timeout = setTimeout(() => controller.abort(), 20000);
    apiRequest<T>(path, { signal: controller.signal })
      .then((data) => {
        if (active) setState({ path, attempt, data, error: null });
      })
      .catch(() => {
        if (active)
          setState({
            path,
            attempt,
            data: null,
            error: "Could not load data. Please try again.",
          });
      })
      .finally(() => clearTimeout(timeout));
    return () => {
      active = false;
      clearTimeout(timeout);
      controller.abort();
    };
  }, [path, attempt]);
  const current =
    state?.path === path && state.attempt === attempt ? state : null;
  return {
    data: current?.data ?? null,
    error: current?.error ?? null,
    retry: () => setAttempt((value) => value + 1),
  };
}
