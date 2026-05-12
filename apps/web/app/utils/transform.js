
export function prepareChartData(f) {
  return f.map(d => ({
    time: d.valid_time,

    median: d.p_50_v,

    // for band
    low: d.p_10_v,
    range: d.p_90_v - d.p_10_v,

    // optional probabilities in %
    prob500: d.prob_v_gt_500 * 100
  }));
}