package main

import (
	"errors"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

const geomagneticProduct = "geomagnetic-activity"
const geomagneticFastProduct = "geomagnetic-fast-l1"
const geomagneticFastArtifact = "kp_ap_fast_l1"

var geomagneticFastSpec = fastForecastSpec{
	product: geomagneticFastProduct, artifact: geomagneticFastArtifact, target: geomagneticProduct,
	interval: 3 * time.Hour, columns: []string{"p_kp_ge_4", "p_kp_ge_5", "p_kp_ge_6", "ap_q10", "ap_q50", "ap_q90"},
	point: decodeFastGeomagnetic,
}

func decodeFastGeomagnetic(values []float64) (fastForecastPoint, error) {
	kp, err := fastThresholds([]float64{4, 5, 6}, values[:3])
	if err != nil {
		return nil, err
	}
	q10, q50, q90 := values[3], values[4], values[5]
	if q10 < 0 || q10 > q50 || q50 > q90 || q90 > 400 {
		return nil, errors.New("unordered fast ap distribution")
	}
	return fastForecastPoint{
		"kp": kp,
		"ap": {Continuous: &publicforecast.QuantileForecast{Q10: q10, Q50: q50, Q90: q90}, Binary: []publicforecast.BinaryForecast{}},
	}, nil
}
