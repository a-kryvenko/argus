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

type FastL1Scores = {
  kp4_brier: number;
  kp5_brier: number;
  kp6_brier: number;
  ap_mae: number;
  ap_coverage_80: number;
  ap_interval_width_80: number;
  alert_threshold: number;
  hits: number;
  false_alerts: number;
};

export type FastL1Evaluation = {
  period: string;
  sample: string;
  notes: string[];
  by_delay: Array<{
    delay_minutes: number;
    targets: number;
    updates: number;
    observed_events: number;
    baseline: FastL1Scores;
    candidate: FastL1Scores;
    ap_mae_delta_ci95?: [number, number];
    kp5_brier_delta_ci95?: [number, number];
  }>;
};

export type ForecastMetrics = {
  evaluation?: { label: string; period: string; sample: string; notes: string[] } | null;
  fast_l1_evaluation?: FastL1Evaluation | null;
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

