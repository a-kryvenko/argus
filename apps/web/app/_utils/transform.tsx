
export function prepareWindChartData(f: Array<any>) : Array<any> {
  return f.map(d => ({
    time: d.valid_time,
    median: parseInt(d.v_q50),
    low: parseInt(d.v_q10),
    high: parseInt(d.v_q90),
  }));
}

export function prepareProbabilitiesData(f: Array<any>) : Array<any> {
  return f.map(d => ({
    time: d.valid_time,
    p_450: parseFloat(d.p_v_ge_450),
    p_500: parseFloat(d.p_v_ge_500),
    p_600: parseFloat(d.p_v_ge_600),
  }));
}