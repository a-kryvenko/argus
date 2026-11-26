package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

const geomagneticProduct = "geomagnetic-activity"
const geomagneticFastProduct = "geomagnetic-fast-l1"
const geomagneticFastArtifact = "kp_ap_fast_l1"
const geomagneticInterval = 3 * time.Hour

type fastGeomagneticRelease struct {
	id                     string
	issueTime, publishedAt time.Time
	points                 map[time.Time]map[string]*publicforecast.VariableForecast
}

func prepareFastGeomagnetic(source release) (fastGeomagneticRelease, error) {
	result := fastGeomagneticRelease{id: source.ID}
	if source.Product != geomagneticFastProduct || source.IssueTime.IsZero() || source.PublishedAt.IsZero() ||
		source.PublishedAt.Before(source.IssueTime) || len(source.Artifacts) != 1 ||
		source.Artifacts[0].Name != geomagneticFastArtifact {
		return result, errors.New("invalid fast geomagnetic release")
	}
	item := source.Artifacts[0]
	columns := []string{"p_kp_ge_4", "p_kp_ge_5", "p_kp_ge_6", "ap_q10", "ap_q50", "ap_q90"}
	if !slices.Equal(item.Columns, append([]string{"issue_time", "valid_time"}, columns...)) {
		return result, errors.New("invalid fast geomagnetic columns")
	}
	rows, err := forecastRows(item, source.IssueTime)
	if err != nil || len(rows) > 2 {
		return result, errors.New("invalid fast geomagnetic rows")
	}
	points := make(map[time.Time]map[string]*publicforecast.VariableForecast, len(rows))
	for _, row := range rows {
		target, err := publicforecast.ParseTimestamp(row["valid_time"])
		if err != nil || !target.Equal(target.Truncate(geomagneticInterval)) ||
			target.Before(source.IssueTime) || target.Add(geomagneticInterval).After(source.IssueTime.Add(6*time.Hour)) ||
			points[target] != nil {
			return result, errors.New("invalid fast geomagnetic target")
		}
		values := make([]float64, len(columns))
		for i, column := range columns {
			value, err := strconv.ParseFloat(row[column], 64)
			if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
				return result, fmt.Errorf("invalid fast geomagnetic value for %s", column)
			}
			values[i] = value
		}
		p4, p5, p6, q10, q50, q90 := values[0], values[1], values[2], values[3], values[4], values[5]
		if p4 > 1 || p4 < p5 || p5 < p6 || p6 < 0 ||
			q10 < 0 || q10 > q50 || q50 > q90 || q90 > 400 {
			return result, errors.New("unordered fast geomagnetic distribution")
		}
		points[target] = map[string]*publicforecast.VariableForecast{
			"kp": {Binary: []publicforecast.BinaryForecast{
				{Threshold: 4, Probability: p4}, {Threshold: 5, Probability: p5}, {Threshold: 6, Probability: p6},
			}},
			"ap": {Continuous: &publicforecast.QuantileForecast{Q10: q10, Q50: q50, Q90: q90},
				Binary: []publicforecast.BinaryForecast{}},
		}
	}
	return fastGeomagneticRelease{result.id, source.IssueTime.UTC(), source.PublishedAt.UTC(), points}, nil
}

type fastGeomagneticProjection struct {
	baseID, updateID     string
	publishedAt, expires time.Time
	body                 []byte
}

// A six-hour window contains at most one native three-hour interval strictly
// after publication, because the interval must also finish within the window.
func projectFastGeomagnetic(base preparedForecast, update fastGeomagneticRelease) (fastGeomagneticProjection, error) {
	result := fastGeomagneticProjection{baseID: base.releaseID, updateID: update.id, publishedAt: update.publishedAt}
	if base.geomagnetic == nil || len(update.points) == 0 {
		return result, nil
	}
	for i, point := range base.geomagnetic.Predictions {
		replacement := update.points[point.Valid.UTC()]
		if replacement == nil || !update.publishedAt.Before(point.Valid) ||
			point.Variables["kp"] == nil || point.Variables["ap"] == nil {
			continue
		}
		forecast := *base.geomagnetic
		forecast.Predictions = slices.Clone(forecast.Predictions)
		copy := *point
		copy.Variables = maps.Clone(point.Variables)
		copy.Variables["kp"], copy.Variables["ap"] = replacement["kp"], replacement["ap"]
		copy.UpdateIssuedAt = &update.issueTime
		forecast.Predictions[i] = &copy
		body, err := json.Marshal(forecast)
		if err != nil {
			return result, err
		}
		result.body = body
		result.expires = point.Valid
		return result, nil
	}
	return result, nil
}

func (p fastGeomagneticProjection) bodyAt(base preparedForecast, now time.Time) []byte {
	if p.body != nil && p.baseID == base.releaseID && !now.Before(p.publishedAt) && now.Before(p.expires) {
		return p.body
	}
	return base.body
}
