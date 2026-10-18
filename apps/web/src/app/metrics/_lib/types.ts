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
  evaluation?: { label: string; period: string; sample: string; notes: string[] } | null;
  target: string;
  meta?: { variables: Record<string, { unit: string }> };
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

