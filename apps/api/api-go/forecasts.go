package main

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

func parseTimestamp(value string) (time.Time, error) {
	// Historical CSVs may separate the date and time with a space.
	if len(value) > 10 && value[10] == ' ' {
		value = value[:10] + "T" + value[11:]
	}
	at, err := time.Parse(time.RFC3339Nano, value)
	return at.UTC(), err
}

func (a *api) summary(w http.ResponseWriter, r *http.Request, respond func(Published[map[string]any])) {
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/summary", nil, nil, 10*time.Second, reportResponseLimit)
	respondPublication(w, respond, raw, err, "Summary is unavailable", func(data map[string]any) bool {
		stamp, hasTimestamp := data["generated_at"].(string)
		_, timeErr := parseTimestamp(stamp)
		_, hasObservations := data["observations"].(map[string]any)
		return hasTimestamp && timeErr == nil && hasObservations
	})
}

func (a *api) solarWindEvents(w http.ResponseWriter, r *http.Request, respond func(Published[publicforecast.SolarWindEventForecast])) {
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/public-forecasts/solar-wind-events", nil, nil, 10*time.Second, reportResponseLimit)
	respondPublication(w, respond, raw, err, "Solar-wind event forecast is not ready",
		func(report publicforecast.SolarWindEventForecast) bool {
			return report.Product == "solar-wind-events" && report.SchemaVersion == 1 && !report.IssueTime.IsZero() &&
				report.SourceStatus != nil && report.Events != nil && report.L1Detections != nil
		})
}

func (a *api) verification(w http.ResponseWriter, r *http.Request, respond func(Published[ForecastVerification])) {
	target := r.PathValue(targetParameter)
	if _, ok := publicforecast.Products[target]; !ok {
		writeError(w, failure{404, "Forecast product not found"})
		return
	}
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/forecasts/"+target+"/verification", nil, nil, 20*time.Second, reportResponseLimit)
	respondPublication(w, respond, raw, err, "Forecast verification is unavailable", func(report ForecastVerification) bool {
		return report.Product == target && !report.Start.IsZero() && report.End.After(report.Start) && report.Groups != nil
	})
}

func (a *api) forecast(w http.ResponseWriter, r *http.Request, respond func(Published[publicforecast.Forecast])) {
	target := r.PathValue(targetParameter)
	product, ok := publicforecast.Products[target]
	if !ok {
		writeError(w, failure{404, "Forecast product not found"})
		return
	}
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/public-forecasts/"+target, nil, nil, 10*time.Second, reportResponseLimit)
	respondPublication(w, respond, raw, err, "Forecast is not ready", func(report publicforecast.Forecast) bool {
		return report.Target == target && validForecast(report, product)
	})
}

func (a *api) density(w http.ResponseWriter, r *http.Request, respond func(Published[publicforecast.DensityForecast])) {
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/public-forecasts/atmospheric-density", nil, nil, 10*time.Second, densityResponseLimit)
	respondPublication(w, respond, raw, err, "Atmospheric density forecast is not ready", validDensity)
}

func (a *api) metrics(w http.ResponseWriter, r *http.Request, respond func(Published[publicforecast.ForecastMetrics])) {
	target := r.PathValue(targetParameter)
	if _, ok := publicforecast.Products[target]; !ok {
		writeError(w, failure{404, "Forecast product not found"})
		return
	}
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/public-forecast-metrics/"+target, nil, nil, 10*time.Second, reportResponseLimit)
	respondPublication(w, respond, raw, err, "Metrics are not ready", func(report publicforecast.ForecastMetrics) bool {
		return report.Target == target && report.Variables != nil
	})
}

// Validate the publication at the gateway boundary and preserve the producer's
// JSON so newer optional fields and precision survive forwarding.
func respondPublication[T any](w http.ResponseWriter, respond func(Published[T]), raw json.RawMessage,
	err error, message string, valid func(T) bool,
) {
	w.Header().Set("Cache-Control", "no-store")
	payload := Published[T]{raw}
	var data *T
	if err != nil || json.Unmarshal(payload.raw, &data) != nil || data == nil || !valid(*data) {
		writeError(w, unavailable(message))
		return
	}
	respond(payload)
}

func validForecast(report publicforecast.Forecast, product publicforecast.Product) bool {
	if report.IssueTime.IsZero() || len(report.Predictions) == 0 || len(report.AvailableVariables) == 0 {
		return false
	}
	variables := make(map[string]bool, len(product.Variables))
	for _, variable := range product.Variables {
		variables[variable.Name] = true
	}
	available := make(map[string]bool, len(report.AvailableVariables))
	for _, name := range report.AvailableVariables {
		if !variables[name] || available[name] {
			return false
		}
		available[name] = true
	}
	previous := -1
	for _, point := range report.Predictions {
		if point == nil || point.Lead <= previous || point.Lead > product.Horizon ||
			!point.Valid.Equal(report.IssueTime.Add(time.Duration(point.Lead)*time.Hour)) || len(point.Variables) == 0 {
			return false
		}
		for name, values := range point.Variables {
			if !available[name] || values == nil {
				return false
			}
			for _, item := range values.Binary {
				if item.Probability < 0 || item.Probability > 1 {
					return false
				}
			}
		}
		previous = point.Lead
	}
	return previous == report.HorizonHours
}

func validDensity(report publicforecast.DensityForecast) bool {
	const horizon = 48
	if report.Target != "atmospheric-density" || report.DriverMode != "observed_persistence" ||
		report.IssueTime.IsZero() || report.ObservedAt.IsZero() || report.ObservedAt.After(report.IssueTime) ||
		report.HorizonHours != horizon || len(report.Predictions) != horizon+1 {
		return false
	}
	size := len(report.Predictions[0].Cells)
	if size == 0 {
		return false
	}
	for lead, point := range report.Predictions {
		if point.Lead != lead || !point.Valid.Equal(report.IssueTime.Add(time.Duration(lead)*time.Hour)) || len(point.Cells) != size {
			return false
		}
		for _, cell := range point.Cells {
			if cell.RhoKgM3 <= 0 || cell.RhoLonP10KgM3 <= 0 || cell.RhoLonP90KgM3 < cell.RhoLonP10KgM3 {
				return false
			}
		}
	}
	return true
}
