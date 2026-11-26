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
	w.Header().Set("Cache-Control", "no-store")
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/summary", nil, nil, 10*time.Second)
	payload := Published[map[string]any]{raw}
	var data map[string]json.RawMessage
	if err != nil || json.Unmarshal(raw, &data) != nil || data == nil {
		writeError(w, unavailable("Summary is unavailable"))
		return
	}
	respond(payload)
}

func (a *api) solarWindEvents(w http.ResponseWriter, r *http.Request, respond func(Published[publicforecast.SolarWindEventForecast])) {
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/public-forecasts/solar-wind-events", nil, nil, 10*time.Second)
	if err != nil {
		writeError(w, unavailable("Solar-wind event forecast is not ready"))
		return
	}
	respond(Published[publicforecast.SolarWindEventForecast]{raw})
}

func (a *api) verification(w http.ResponseWriter, r *http.Request, respond func(Published[ForecastVerification])) {
	target := r.PathValue(targetParameter)
	if _, ok := publicforecast.Products[target]; !ok {
		writeError(w, failure{404, "Forecast product not found"})
		return
	}
	raw, _, err := a.request(r.Context(), a.prophet, "GET", "/internal/v1/forecasts/"+target+"/verification", nil, nil, 20*time.Second)
	payload := Published[ForecastVerification]{raw}
	var report ForecastVerification
	if err != nil || payload.decode(&report) != nil || report.Product != target ||
		report.Start.IsZero() || report.End.Before(report.Start) || report.Groups == nil {
		writeError(w, unavailable("Forecast verification is unavailable"))
		return
	}
	respond(payload)
}

func (a *api) forecast(w http.ResponseWriter, r *http.Request, respond func(Published[publicforecast.Forecast])) {
	target := r.PathValue(targetParameter)
	if _, ok := publicforecast.Products[target]; !ok {
		writeError(w, failure{404, "Forecast product not found"})
		return
	}
	raw, err := a.projectedForecast(r, "/internal/v1/public-forecasts/"+target, "Forecast is not ready")
	if err != nil {
		writeError(w, err)
		return
	}
	respond(Published[publicforecast.Forecast]{raw})
}

func (a *api) density(w http.ResponseWriter, r *http.Request, respond func(Published[publicforecast.DensityForecast])) {
	raw, err := a.projectedForecast(r, "/internal/v1/public-forecasts/atmospheric-density", "Atmospheric density forecast is not ready")
	if err != nil {
		writeError(w, err)
		return
	}
	respond(Published[publicforecast.DensityForecast]{raw})
}

func (a *api) projectedForecast(r *http.Request, path, message string) (json.RawMessage, error) {
	raw, _, err := a.request(r.Context(), a.prophet, "GET", path, nil, nil, 10*time.Second)
	if err != nil {
		return nil, unavailable(message)
	}
	return raw, nil
}

func (a *api) metrics(w http.ResponseWriter, r *http.Request, respond func(Published[publicforecast.ForecastMetrics])) {
	target := r.PathValue(targetParameter)
	if _, ok := publicforecast.Products[target]; !ok {
		writeError(w, failure{404, "Forecast product not found"})
		return
	}
	raw, err := a.projectedForecast(r, "/internal/v1/public-forecast-metrics/"+target, "Metrics are not ready")
	if err != nil {
		writeError(w, err)
		return
	}
	respond(Published[publicforecast.ForecastMetrics]{raw})
}
