
const now = new Date();

export const forecast = Array.from({ length: 96 }, (_, i) => ({
  valid_time: new Date(now.getTime() + i * 60 * 60 * 1000).toISOString(),
  mean_v: 450 + Math.random() * 200,
  std_v: Math.random() * 50,
  p_10_v: 400 + Math.random() * 100,
  p_50_v: 450 + Math.random() * 200,
  p_90_v: 600 + Math.random() * 200,
  prob_v_gt_500: Math.random(),
  prob_v_gt_600: Math.random(),
  prob_v_gt_700: Math.random(),
  kp_risk_proxy: Math.random(),
  drag_risk_proxy: Math.random()
}));