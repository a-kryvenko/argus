
export function prepareChartData(forecast) {
  return forecast.map(d => ({
    time: d.time,

    median: d.p50_v,

    // for band
    low: d.p10_v,
    range: d.p90_v - d.p10_v,

    // optional probabilities in %
    prob500: d.prob_v_gt_500 * 100
  }));
}