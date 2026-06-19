
export function prepareWindChartData(f: Array<any>) : Array<any> {
  return f.map(d => ({
    time: d.valid_time,
    median: parseInt(d.v_q50),
    low: parseInt(d.v_q10),
    high: parseInt(d.v_q90),
  }));
}

export function preparePlasmaHeatmapData(f: Array<any>) : Array<Array<Number>> {
  const riskData: Array<Array<Number>> = []
  if (f.length > 0) {
    for (let i = 0; i < f.length; i ++) {
      riskData.push([i, 0, Math.floor(f[i]["p_v_ge_450"] * 100)])
      riskData.push([i, 1, Math.floor(f[i]["p_v_ge_500"] * 100)])
      riskData.push([i, 2, Math.floor(f[i]["p_v_ge_600"] * 100)])
    }
  }

  return riskData;
}

export function prepareKpHeatmapData(f: Array<any>): Array<Array<Number>> {
  const riskData: Array<Array<Number>> = []

  for (let i = 0; i < f.length; i ++) {
    riskData.push([i, 0, Math.floor(f[i]["p_kp_4"] * 100)])
    riskData.push([i, 1, Math.floor(f[i]["p_kp_5"] * 100)])
    riskData.push([i, 2, Math.floor(f[i]["p_kp_6"] * 100)])
  }

  return riskData;
}