export function prepareTotalHmfHeatmapData(f: Array<any>): Array<Array<Number>> {
  const riskData: Array<Array<Number>> = []

  for (let i = 0; i < f.length; i ++) {
    riskData.push([i, 0, Math.floor(f[i]["p_bt_ge_5"] * 100)])
    riskData.push([i, 1, Math.floor(f[i]["p_bt_ge_10"] * 100)])
    riskData.push([i, 2, Math.floor(f[i]["p_bt_ge_15"] * 100)])
  }

  return riskData;
}

export function prepareSouthwardHmfHeatmapData(f: Array<any>): Array<Array<Number>> {
  const riskData: Array<Array<Number>> = []

  for (let i = 0; i < f.length; i ++) {
    riskData.push([i, 0, Math.floor(f[i]["p_southward_bz_ge_5"] * 100)])
    riskData.push([i, 1, Math.floor(f[i]["p_southward_bz_ge_10"] * 100)])
    riskData.push([i, 2, Math.floor(f[i]["p_southward_bz_ge_15"] * 100)])
  }

  return riskData;
}