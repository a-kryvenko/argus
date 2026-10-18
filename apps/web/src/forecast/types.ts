export type BinaryForecast = {
  threshold: number;
  operator?: "gte";
  probability: number;
};

export type ForecastPoint = {
  valid_time: string;
  lead_hours: number;
  variables: Record<string, {
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
  meta?: { variables: Record<string, { unit: string }> };
  cme_speed?: {
    integrated_into_speed?: boolean;
    state: 'available' | 'partial' | 'unavailable';
    as_of?: string | null;
    scenario_kind: 'conditional_scenarios_not_quantiles';
    events: { event_id: string; peak_speed: number; arrival_at: string }[];
    points: { valid_time: string; speed_background: number; speed_nominal: number;
      speed_early: number; speed_late: number; scenario_lower: number; scenario_upper: number; speed_integrated?: number | null }[];
  } | null;
};

