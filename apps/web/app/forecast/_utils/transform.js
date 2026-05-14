
export function prepareChartData(f) {
  return f.map(d => ({
    time: d.valid_time,
    median: parseInt(d.p_50_v),
    low: parseInt(d.p_10_v),
    high: parseInt(d.p_90_v),
  }));
}