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

const fastForecastHorizon = 6 * time.Hour

type fastForecastPoint map[string]*publicforecast.VariableForecast

type fastForecastSpec struct {
	product, artifact, target string
	interval                  time.Duration
	columns                   []string
	point                     func([]float64) (fastForecastPoint, error)
}

var fastForecastSpecs = []fastForecastSpec{geomagneticFastSpec, hmfFastSpec}

func fastForecastSpecFor(product string) *fastForecastSpec {
	for i := range fastForecastSpecs {
		if fastForecastSpecs[i].product == product {
			return &fastForecastSpecs[i]
		}
	}
	return nil
}

type fastForecastRelease struct {
	id                     string
	issueTime, publishedAt time.Time
	points                 map[time.Time]fastForecastPoint
}

func (spec fastForecastSpec) prepare(source release) (fastForecastRelease, error) {
	result := fastForecastRelease{id: source.ID}
	if source.Product != spec.product || source.IssueTime.IsZero() || source.PublishedAt.IsZero() ||
		source.PublishedAt.Before(source.IssueTime) || len(source.Artifacts) != 1 ||
		source.Artifacts[0].Name != spec.artifact {
		return result, errors.New("invalid fast release")
	}
	item := source.Artifacts[0]
	if !slices.Equal(item.Columns, append([]string{"issue_time", "valid_time"}, spec.columns...)) {
		return result, errors.New("invalid fast release columns")
	}
	rows, err := forecastRows(item, source.IssueTime)
	if err != nil || len(rows) > int(fastForecastHorizon/spec.interval) {
		return result, errors.New("invalid fast release rows")
	}
	points := make(map[time.Time]fastForecastPoint, len(rows))
	for _, row := range rows {
		target, err := publicforecast.ParseTimestamp(row["valid_time"])
		if err != nil || !target.Equal(target.Truncate(spec.interval)) || target.Before(source.IssueTime) ||
			target.Add(spec.interval).After(source.IssueTime.Add(fastForecastHorizon)) || points[target] != nil {
			return result, errors.New("invalid fast release target")
		}
		values := make([]float64, len(spec.columns))
		for i, column := range spec.columns {
			value, err := strconv.ParseFloat(row[column], 64)
			if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
				return result, fmt.Errorf("invalid fast release value for %s", column)
			}
			values[i] = value
		}
		point, err := spec.point(values)
		if err != nil {
			return result, err
		}
		points[target] = point
	}
	return fastForecastRelease{result.id, source.IssueTime.UTC(), source.PublishedAt.UTC(), points}, nil
}

func fastThresholds(thresholds, values []float64) (*publicforecast.VariableForecast, error) {
	probabilities := make([]publicforecast.BinaryForecast, len(thresholds))
	upper := 1.0
	for i, value := range values {
		if value < 0 || value > upper {
			return nil, errors.New("unordered fast threshold probabilities")
		}
		probabilities[i] = publicforecast.BinaryForecast{Threshold: thresholds[i], Probability: value}
		upper = value
	}
	return &publicforecast.VariableForecast{Binary: probabilities}, nil
}

type fastForecastProjection struct {
	baseID, updateID string
	publishedAt      time.Time
	bodies           []fastForecastBody
}

type fastForecastBody struct {
	expires time.Time
	content []byte
}

func projectFastForecast(base preparedForecast, update fastForecastRelease) (fastForecastProjection, error) {
	result := fastForecastProjection{baseID: base.releaseID, updateID: update.id, publishedAt: update.publishedAt}
	if base.forecast == nil || len(update.points) == 0 {
		return result, nil
	}
	forecast := *base.forecast
	forecast.Predictions = slices.Clone(forecast.Predictions)
	// Cache a response for each expiry so requests can use the current time
	// without re-encoding the forecast or waiting for a database notification.
	for i := len(forecast.Predictions) - 1; i >= 0; i-- {
		point := forecast.Predictions[i]
		replacement := update.points[point.Valid.UTC()]
		if len(replacement) == 0 || !update.publishedAt.Before(point.Valid) ||
			!hasFastVariables(point.Variables, replacement) {
			continue
		}
		next := *point
		next.Variables = maps.Clone(point.Variables)
		if len(replacement) > 1 {
			next.UpdateIssuedAt = &update.issueTime
		}
		for variable, value := range replacement {
			copy := *value
			if len(replacement) == 1 {
				copy.UpdateIssuedAt = &update.issueTime
			}
			next.Variables[variable] = &copy
		}
		forecast.Predictions[i] = &next
		body, err := json.Marshal(forecast)
		if err != nil {
			return fastForecastProjection{}, err
		}
		result.bodies = append(result.bodies, fastForecastBody{point.Valid, body})
	}
	return result, nil
}

func hasFastVariables(variables map[string]*publicforecast.VariableForecast, replacement fastForecastPoint) bool {
	for variable := range replacement {
		if variables[variable] == nil {
			return false
		}
	}
	return true
}

func (p fastForecastProjection) bodyAt(base preparedForecast, now time.Time) []byte {
	if p.baseID == base.releaseID && !now.Before(p.publishedAt) {
		for i := len(p.bodies) - 1; i >= 0; i-- {
			if now.Before(p.bodies[i].expires) {
				return p.bodies[i].content
			}
		}
	}
	return base.body
}
