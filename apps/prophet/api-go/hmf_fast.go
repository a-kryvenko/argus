package main

import "time"

const hmfProduct = "hmf"
const hmfFastProduct = "hmf-fast-l1"
const hmfFastArtifact = "bt_fast_l1"

var hmfFastSpec = fastForecastSpec{
	product: hmfFastProduct, artifact: hmfFastArtifact, target: hmfProduct,
	interval: time.Hour, columns: []string{"p_bt_ge_5", "p_bt_ge_10", "p_bt_ge_15"},
	point: decodeFastBt,
}

func decodeFastBt(values []float64) (fastForecastPoint, error) {
	bt, err := fastThresholds([]float64{5, 10, 15}, values)
	if err != nil {
		return nil, err
	}
	return fastForecastPoint{"bt": bt}, nil
}
