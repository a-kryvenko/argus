export type DragInputs = {
  altitude_km: number;
  inclination_deg: number;
  mass_kg: number;
  effective_area_m2: number;
  drag_coefficient: number;
  horizon_hours: number;
};

export type DragPoint = {
  valid_time: string;
  lead_hours: number;
  mean_density_kg_m3: number;
  mean_drag_accel_m_s2: number;
  delta_v_loss_m_s: number;
  estimated_altitude_loss_m: number;
};

export type DragAssessment = Omit<DragPoint, "valid_time" | "lead_hours"> & {
  computed_at: string;
  start_time: string;
  end_time: string;
  inputs: DragInputs;
  source: {
    release_id: string;
    issue_time: string;
    observed_at: string;
    dtc_observed_at: string;
    driver_mode: 'observed_persistence';
    background_interpolated: boolean;
  };
  meta?: { model: string; source: { model: string } };
  assumptions: string[];
  predictions: DragPoint[];
};

export function utc(value: string) {
  return new Date(value).toLocaleString("en-GB", { timeZone: "UTC", hour12: false });
}

export function number(value: number) {
  return value === 0 ? "0" : Math.abs(value) < 0.01
    ? value.toExponential(3)
    : value.toLocaleString("en-GB", { maximumFractionDigits: 3 });
}
