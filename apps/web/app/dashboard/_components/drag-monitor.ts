import type { DragPoint } from './leo-drag';

export type AttitudeMode = { name: string; effective_area_m2: number; drag_coefficient: number };
export type ProfileInput = {
  name: string; altitude_km: number; inclination_deg: number; mass_kg: number;
  horizon_hours: 24 | 48;
  thresholds: { elevated_altitude_loss_m: number; high_altitude_loss_m: number };
  modes: AttitudeMode[]; active_mode: string; enabled: boolean;
  event_policy?: { growth_percent: number; growth_min_m: number } | null;
};
export type Profile = ProfileInput & { id: string; revision: number; updated_at: string };
export type Assessment = {
  start_time: string; end_time: string; computed_at: string;
  estimated_altitude_loss_m: number; delta_v_loss_m_s: number;
  drag_risk: 'not_assessed' | 'low' | 'elevated' | 'high'; risk_reason: string;
  source: { density_model: string; release_id: string; observed_at: string;
    geomagnetic_observed_at: string | null; driver_mode: string };
  assumptions: string[]; predictions: DragPoint[];
};
export type Run = {
  id: string; profile_id: string; profile_revision: number; profile_snapshot: ProfileInput;
  assessed_at: string; release_id: string | null;
  modes: { mode: string; assessment: Assessment | null; error: string | null;
    elevated_at: string | null; high_at: string | null }[];
};
export type MonitorItem = {
  profile: Profile; latest: Run | null;
  data_status: 'not_assessed' | 'profile_changed' | 'unavailable' | 'stale' | 'current';
  attention: 'not_assessed' | 'low' | 'elevated' | 'high'; loss_change_m: number | null;
};
export type ChartPoint = { lead_hours: number; valid_time: string; altitude_loss_m: number };
export type OperatorChart = {
  run_id: string; prepared_at: string; start_time: string | null; end_time: string | null;
  horizon_hours: 24 | 48; now_position: 'before' | 'within' | 'after' | 'unavailable';
  now_lead_hours: number | null; elapsed_until_hours: number | null; max_loss_m: number;
  data_status: MonitorItem['data_status']; historical: boolean; thresholds: ProfileInput['thresholds'];
  modes: {
    mode: string; active: boolean; error: 'unavailable' | 'outside_model_domain' | null;
    points: ChartPoint[]; elapsed: ChartPoint[]; remaining: ChartPoint[]; previous: ChartPoint[];
    previous_issue_time: string | null; previous_release_id: string | null; loss_change_m: number | null;
    crossings: { level: 'elevated' | 'high'; lead_hours: number; valid_time: string;
      altitude_loss_m: number; state: 'elapsed' | 'upcoming' }[];
  }[];
};
export const dataLabels = {
  not_assessed: 'Awaiting assessment', profile_changed: 'Profile changed',
  unavailable: 'Assessment unavailable', stale: 'Stale atmosphere', current: 'Current atmosphere',
};
export const attentionLabels = { not_assessed: 'Not assessed', low: 'Within limits', elevated: 'Review required', high: 'High limit reached' };

export type DragEvent = {
  id: string; profile_id: string; profile_revision: number; spacecraft_name: string; active_mode: string;
  kind: 'risk_elevated' | 'risk_high' | 'risk_reduced' | 'risk_recovered' | 'loss_increased' | 'data_stale' | 'assessment_unavailable' | 'data_recovered';
  severity: 'info' | 'warning' | 'high'; detected_at: string; expected_at: string | null;
  run_id: string; release_id: string | null; comparison_run_id: string | null;
  message: string; values: Record<string, number | string | null>;
  acknowledged_at: string | null; acknowledged_by: number | null;
};
export type DragEventPage = { items: DragEvent[]; total: number; unacknowledged: number };
