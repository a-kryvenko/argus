package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"
)

var (
	windMetrics   = []string{"bx", "by", "bz", "bt", "v", "n", "t"}
	indexMetrics  = []string{"kp", "dst"}
	hourlyMetrics = []string{"ap", "f10_7", "s10", "m10", "y10"}
	allMetrics    = slices.Concat(windMetrics, indexMetrics, hourlyMetrics)
	indexDuration = map[string]time.Duration{"kp": 3 * time.Hour, "dst": time.Hour, "ap": 3 * time.Hour}
)

type sample struct {
	at    time.Time
	value *float64
}

func (a *api) numeric(ctx context.Context, resource string, metrics []string, from, to time.Time, interval string) (map[string][]sample, error) {
	query := url.Values{"metrics": {strings.Join(metrics, ",")}}
	if resource == "history" {
		query.Set("from", from.Format(time.RFC3339Nano))
		query.Set("to", to.Format(time.RFC3339Nano))
		query.Set("interval", interval)
	}
	raw, _, err := a.request(ctx, a.clio, "GET", "/internal/v1/observations/"+resource, query, nil, 90*time.Second)
	if err != nil {
		return nil, unavailable("Observation service is unavailable")
	}
	var result struct {
		Success bool `json:"success"`
		Data    struct {
			Points []map[string]json.RawMessage `json:"points"`
		} `json:"data"`
	}
	if json.Unmarshal(raw, &result) != nil || !result.Success || result.Data.Points == nil {
		return nil, unavailable("Observation service returned an invalid response")
	}
	series := make(map[string][]sample, len(metrics))
	var previous time.Time
	for _, row := range result.Data.Points {
		var at time.Time
		if json.Unmarshal(row["time"], &at) != nil || at.IsZero() || (previous.After(at)) {
			return nil, unavailable("Observation service returned an invalid response")
		}
		previous = at
		for _, metric := range metrics {
			encoded, found := row[metric]
			if !found {
				continue
			}
			var value *float64
			if json.Unmarshal(encoded, &value) != nil {
				return nil, unavailable("Observation service returned an invalid response")
			}
			series[metric] = append(series[metric], sample{at.UTC(), value})
		}
	}
	return series, nil
}

func observationSelection(query url.Values) ([]string, error) {
	if values := query[observationsParameter]; len(values) > 0 {
		selected := []string{}
		for _, value := range strings.Split(values[len(values)-1], ",") {
			metric := strings.TrimSpace(value)
			if !slices.Contains(allMetrics, metric) {
				return nil, invalid("Unknown observation. Supported: " + strings.Join(allMetrics, ","))
			}
			if !slices.Contains(selected, metric) {
				selected = append(selected, metric)
			}
		}
		return selected, nil
	}
	return allMetrics, nil
}

func selectedMetrics(selected, group []string) []string {
	var result []string
	for _, metric := range selected {
		if slices.Contains(group, metric) {
			result = append(result, metric)
		}
	}
	return result
}

func quality(point sample) *string {
	if point.value == nil {
		return pointer("missing")
	}
	return nil
}

func pointer[T any](value T) *T { return &value }

func (data *CurrentObservations) setSample(metric string, points []sample) {
	if slices.Contains(indexMetrics, metric) {
		var index *CurrentIndexSample
		if len(points) > 0 {
			point := points[len(points)-1]
			index = &CurrentIndexSample{IntervalStart: point.at, IntervalEnd: point.at.Add(indexDuration[metric]),
				Value: point.value, Quality: quality(point)}
		}
		if metric == "kp" {
			data.Kp = &index
		} else {
			data.Dst = &index
		}
		return
	}
	var solar *CurrentSolarSample
	if len(points) > 0 {
		point := points[len(points)-1]
		solar = &CurrentSolarSample{ObservedAt: point.at, Value: point.value, Quality: quality(point)}
		if metric == "ap" {
			solar.IntervalEnd = pointer(point.at.Add(indexDuration[metric]))
		}
	}
	switch metric {
	case "bx":
		data.Bx = &solar
	case "by":
		data.By = &solar
	case "bz":
		data.Bz = &solar
	case "bt":
		data.Bt = &solar
	case "v":
		data.V = &solar
	case "n":
		data.N = &solar
	case "t":
		data.T = &solar
	case "ap":
		data.AP = &solar
	case "f10_7":
		data.F107 = &solar
	case "s10":
		data.S10 = &solar
	case "m10":
		data.M10 = &solar
	case "y10":
		data.Y10 = &solar
	}
}

func (a *api) currentObservations(w http.ResponseWriter, r *http.Request, respond func(CurrentObservations)) {
	w.Header().Set("Cache-Control", "no-store")
	query := r.URL.Query()
	selected, err := observationSelection(query)
	if err != nil {
		writeError(w, err)
		return
	}
	meta, err := booleanQuery(query, metaParameter)
	if err != nil {
		writeError(w, err)
		return
	}
	trends, err := booleanQuery(query, trendsParameter)
	if err != nil {
		writeError(w, err)
		return
	}
	now := time.Now().UTC()
	current, err := a.numeric(r.Context(), "latest", selected, time.Time{}, time.Time{}, "")
	if err != nil {
		writeError(w, err)
		return
	}
	data := CurrentObservations{}
	series := make(map[string]SeriesMetadata)
	for _, metric := range selected {
		data.setSample(metric, current[metric])
		if meta && !slices.Contains(hourlyMetrics, metric) {
			series[metric] = a.observationMetadata(metric, "1m")
		}
	}
	if len(series) > 0 {
		data.Meta = &ObservationMetadata{Series: series}
	}
	if trends {
		changes, err := a.observationChanges(r.Context(), current, selected, now)
		if err != nil {
			writeError(w, err)
			return
		}
		data.Changes1h = changes
	}
	respond(data)
}

func historyQuery(query url.Values, now time.Time) (from, to time.Time, resolution string, limit int, bounded bool, err error) {
	fromInput, toInput := query[fromParameter], query[toParameter]
	bounded = len(fromInput) > 0 || len(toInput) > 0
	to = now
	if len(toInput) > 0 {
		to, err = parseTimestamp(toInput[len(toInput)-1])
		if err != nil {
			err = invalid("from and to must include a timezone and define a positive interval")
			return
		}
	}
	from = to.Add(-24 * time.Hour)
	if len(fromInput) > 0 {
		from, err = parseTimestamp(fromInput[len(fromInput)-1])
	}
	if err != nil || !from.Before(to) {
		err = invalid("from and to must include a timezone and define a positive interval")
		return
	}
	resolution = defaultResolution
	if values := query[resolutionParameter]; len(values) > 0 {
		resolution = values[len(values)-1]
	}
	if !slices.Contains(observationResolutions, resolution) {
		err = invalid("resolution must be 1m, 5m, 1h or auto")
		return
	}
	if resolution == "auto" {
		switch {
		case to.Sub(from) <= 6*time.Hour:
			resolution = "1m"
		case to.Sub(from) <= 24*time.Hour:
			resolution = "5m"
		default:
			resolution = "1h"
		}
	}
	limit = defaultObservationLimit
	if values := query[limitParameter]; len(values) > 0 {
		limit, err = strconv.Atoi(values[len(values)-1])
		if err != nil || limit < 1 || limit > maxObservationLimit {
			err = invalid("limit must be between 1 and 168")
		}
	}
	return
}

type historyRead struct {
	metrics    []string
	start, end time.Time
	interval   string
}

func (a *api) observationHistory(w http.ResponseWriter, r *http.Request, respond func(ObservationHistory)) {
	w.Header().Set("Cache-Control", "no-store")
	query := r.URL.Query()
	selected, err := observationSelection(query)
	if err != nil {
		writeError(w, err)
		return
	}
	meta, err := booleanQuery(query, metaParameter)
	if err != nil {
		writeError(w, err)
		return
	}
	now := time.Now().UTC()
	from, to, resolution, limit, bounded, err := historyQuery(query, now)
	if err != nil {
		writeError(w, err)
		return
	}
	wind := selectedMetrics(selected, windMetrics)
	indices := selectedMetrics(selected, indexMetrics)
	hourly := selectedMetrics(selected, hourlyMetrics)
	days := map[string]int{"1m": 7, "5m": 31, "1h": 366}[resolution]
	if len(wind) > 0 && to.Sub(from) > time.Duration(days)*24*time.Hour {
		writeError(w, invalid("History interval must be at most "+strconv.Itoa(days)+" days for "+resolution))
		return
	}
	if len(indices) > 0 && to.Sub(from) > 31*24*time.Hour {
		writeError(w, invalid("Geomagnetic history interval must be at most 31 days"))
		return
	}
	metadata := make(map[string]SeriesMetadata)
	data := ObservationHistory{From: from, To: to, Series: make(map[string]HistorySeries)}
	reads := []historyRead{
		{wind, from, to, resolution},
		{indices, from.Add(-3 * time.Hour), to, "1m"},
	}
	hourlyFrom := from
	if !bounded {
		hourlyFrom = to.Add(-time.Duration(limit) * time.Hour)
	}
	if len(hourly) > 0 {
		if to.Sub(hourlyFrom) > 366*24*time.Hour {
			writeError(w, invalid("Use an interval of at most 366 days"))
			return
		}
		reads = append(reads, historyRead{hourly, hourlyFrom, to, "1h"})
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	type result struct {
		series map[string][]sample
		err    error
	}
	results := make([]chan result, len(reads))
	for i, read := range reads {
		if len(read.metrics) == 0 {
			continue
		}
		results[i] = make(chan result, 1)
		go func(out chan<- result, read historyRead) {
			rows, err := a.numeric(ctx, "history", read.metrics, read.start, read.end, read.interval)
			out <- result{rows, err}
		}(results[i], read)
	}
	for i, output := range results {
		if output == nil {
			continue
		}
		part := <-output
		if part.err != nil {
			writeError(w, part.err)
			return
		}
		if i == 2 {
			data.Points = pointer(hourlyPoints(part.series, hourly, hourlyFrom, to, limit))
			continue
		}
		interval := reads[i].interval
		step := resolutionDuration(interval)
		if i == 0 {
			data.ResolutionSeconds = pointer(int(step.Seconds()))
		}
		for _, metric := range reads[i].metrics {
			points := part.series[metric]
			duration := indexDuration[metric]
			if duration > 0 {
				points = overlapping(points, from, to, duration)
				items := make([]IndexSample, 0, len(points))
				for _, point := range points {
					end := point.at.Add(duration)
					status := "completed"
					if end.After(now) {
						status = "in_progress"
					}
					items = append(items, IndexSample{IntervalStart: point.at, IntervalEnd: end,
						IntervalStatus: status, Value: point.value, Quality: quality(point)})
				}
				data.Series[metric] = HistorySeries{Index: &IndexHistorySeries{Points: items,
					Coverage: observationCoverage(points, from, to, now, duration, true), DataStatus: indexDataStatus(metric)}}
			} else {
				items := make([]SolarSample, 0, len(points))
				for _, point := range points {
					items = append(items, SolarSample{ObservedAt: point.at, Value: point.value, Quality: quality(point)})
				}
				data.Series[metric] = HistorySeries{Solar: &SolarHistorySeries{Points: items,
					Coverage: observationCoverage(points, from, to, now, step, false)}}
			}
			metadata[metric] = a.observationMetadata(metric, interval)
		}
		if meta {
			gapFilling := "none"
			if interval == "1h" && i == 0 {
				gapFilling = "normalized"
			}
			data.Meta = &ObservationMetadata{Series: metadata, GapFilling: &gapFilling}
		}
	}
	respond(data)
}

func resolutionDuration(resolution string) time.Duration {
	return map[string]time.Duration{"1m": time.Minute, "5m": 5 * time.Minute, "1h": time.Hour}[resolution]
}

func hourlyPoints(series map[string][]sample, metrics []string, from, to time.Time, limit int) []map[string]any {
	rows := map[time.Time]map[string]any{}
	var times []time.Time
	for _, metric := range metrics {
		for _, point := range series[metric] {
			if point.at.Before(from) || !point.at.Before(to) {
				continue
			}
			row := rows[point.at]
			if row == nil {
				row = map[string]any{"issue_time": point.at}
				for _, selected := range metrics {
					row[selected] = nil
				}
				rows[point.at] = row
				times = append(times, point.at)
			}
			row[metric] = point.value
		}
	}
	slices.SortFunc(times, time.Time.Compare)
	points := make([]map[string]any, 0, min(len(times), limit))
	for _, at := range times[max(0, len(times)-limit):] {
		points = append(points, rows[at])
	}
	return points
}

func overlapping(points []sample, from, to time.Time, duration time.Duration) []sample {
	result := make([]sample, 0, len(points))
	for _, point := range points {
		if point.at.Before(to) && point.at.Add(duration).After(from) {
			result = append(result, point)
		}
	}
	return result
}
