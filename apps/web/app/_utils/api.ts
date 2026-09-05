type ApiErrorBody = {
  code?: string;
  message?: string;
};

export type ObservationPoint = {
  issue_time: string;
  bx: number;
  by: number;
  bz: number;
  v: number;
  n: number;
  t: number;
  kp: number;
  dst: number;
  ap: number;
  f10_7: number;
  s10: number | null;
  m10: number | null;
  y10: number | null;
};

type ApiResponse<T> = {
  success: boolean;
  data: T | null;
  error: ApiErrorBody | null;
};

export type BinaryForecast = {
  threshold: number;
  operator: "gte";
  probability: number;
};

export type ForecastPoint = {
  valid_time: string;
  lead_hours: number;
  variables: Record<string, {
    unit: string;
    continuous: { q10: number; q50: number; q90: number } | null;
    binary: BinaryForecast[];
  }>;
};

export type Forecast = {
  target: string;
  issue_time: string;
  horizon_hours: number;
  available_variables: string[];
  predictions: ForecastPoint[];
};

export type ReliabilityPoint = {
  predicted_probability: number;
  observed_frequency: number;
};

export type BinaryMetricsPoint = {
  lead_hours: number;
  brier_score: number | null;
  roc_auc: number | null;
  average_precision: number | null;
  threat_score: number | null;
  heidke_skill_score: number | null;
  reliability: ReliabilityPoint[];
};

export type ForecastMetrics = {
  target: string;
  variables: Record<string, {
    continuous: {
      quantiles: number[];
      by_lead_hour: Array<{ lead_hours: number; values: Record<string, number | null> }>;
    } | null;
    binary: Array<{
      threshold: number;
      operator: "gte";
      by_lead_hour: BinaryMetricsPoint[];
    }>;
  }>;
};

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getApiUrl(path: string): string {
  const baseUrl = "/api/v1";
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  return `${baseUrl}${normalizedPath}`;
}

export async function apiRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(getApiUrl(path), options);
  let body: ApiResponse<T> | null = null;

  try {
    body = (await response.json()) as ApiResponse<T>;
  } catch {
    throw new ApiError(
      response.ok ? "API returned an invalid JSON response" : `API request failed with status ${response.status}`,
      response.status,
    );
  }

  if (!response.ok || !body.success || body.data === null) {
    throw new ApiError(
      body.error?.message ?? `API request failed with status ${response.status}`,
      response.status,
      body.error?.code,
    );
  }

  return body.data;
}
