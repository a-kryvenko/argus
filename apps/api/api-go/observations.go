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

	"github.com/a-kryvenko/argus/observationcatalog"
)

var (
	windMetrics     = observationcatalog.IDs("wind")
	indexMetrics    = observationcatalog.IDs("index")
	hourlyMetrics   = observationcatalog.IDs("background")
	xrsMetrics      = observationcatalog.IDs("xrs")
	protonMetrics   = observationcatalog.IDs("protons")
	historyDefaults = slices.Concat(windMetrics, indexMetrics, hourlyMetrics)
	allMetrics      = slices.Concat(historyDefaults, xrsMetrics, protonMetrics)
	indexDuration   = func() map[string]time.Duration {
		intervals := make(map[string]time.Duration)
		for _, metric := range observationcatalog.Metrics {
			if metric.SourceInterval {
				intervals[metric.ID] = time.Duration(metric.NativeSeconds) * time.Second
			}
		}
		return intervals
	}()
)

const (
	l1TimeBasis        = "l1"
	nearEarthTimeBasis = "near_earth"
)

type sample = observationcatalog.Point

func (a *api) numeric(ctx context.Context, metrics []string, from, to time.Time, interval, timeBasis string) (map[string]observationcatalog.Series, error) {
	query := url.Values{"metrics": {strings.Join(metrics, ",")}}
	if !from.IsZero() {
		query.Set("from", from.Format(time.RFC3339Nano))
		query.Set("to", to.Format(time.RFC3339Nano))
		query.Set("interval", interval)
	}
	if timeBasis != "" {
		query.Set(timeBasisParameter, timeBasis)
	}
	raw, _, err := a.request(ctx, a.clio, "GET", "/internal/v1/observations/series", query, nil, 90*time.Second)
	if err != nil {
		return nil, unavailable("Observation service is unavailable")
	}
	var result struct {
		Success bool                                 `json:"success"`
		Data    map[string]observationcatalog.Series `json:"data"`
	}
	if json.Unmarshal(raw, &result) != nil || !result.Success || result.Data == nil {
		return nil, unavailable("Observation service returned an invalid response")
	}
	for _, metric := range metrics {
		series := result.Data[metric]
		if series.Points == nil || series.ResolutionSeconds <= 0 || series.TimeBasis == "" ||
			(timeBasis != "" && slices.Contains(windMetrics, metric) && series.TimeBasis != timeBasis) {
			return nil, unavailable("Observation service returned an invalid series")
		}
		var previous time.Time
		for _, point := range series.Points {
			if point.ObservedAt.IsZero() || !point.ObservedAt.After(previous) {
				return nil, unavailable("Observation service returned an invalid series")
			}
			previous = point.ObservedAt
		}
	}
	return result.Data, nil
}

func observationSelection(query url.Values, defaults []string) ([]string, error) {
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
	return defaults, nil
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
	if point.Quality != "" {
		return pointer(point.Quality)
	}
	if point.Value == nil {
		return pointer("missing")
	}
	return nil
}

func provenance(point sample) Provenance {
	return Provenance{SourceProduct: point.SourceProduct, Spacecraft: point.Spacecraft, ReceivedAt: point.ReceivedAt}
}

func pointer[T any](value T) *T { return &value }

func (data *CurrentObservations) setSample(metric string, series observationcatalog.Series) {
	points := series.Points
	if slices.Contains(indexMetrics, metric) {
		var index *CurrentIndexSample
		if len(points) > 0 {
			point := points[len(points)-1]
			index = &CurrentIndexSample{IntervalStart: point.ObservedAt, IntervalEnd: point.ObservedAt.Add(indexDuration[metric]),
				Value: point.Value, Quality: quality(point), Provenance: provenance(point), TimeBasis: series.TimeBasis}
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
		solar = &CurrentSolarSample{ObservedAt: point.ObservedAt, Value: point.Value, Quality: quality(point), Provenance: provenance(point), TimeBasis: series.TimeBasis}
		if metric == "ap" {
			solar.IntervalEnd = pointer(point.ObservedAt.Add(indexDuration[metric]))
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
	case "xrs_a":
		data.XRSA = &solar
	case "xrs_b":
		data.XRSB = &solar
	case "proton_10mev":
		data.Proton10MeV = &solar
	case "proton_100mev":
		data.Proton100MeV = &solar
	}
}

func (a *api) currentObservations(w http.ResponseWriter, r *http.Request, respond func(CurrentObservations)) {
	w.Header().Set("Cache-Control", "no-store")
	query := r.URL.Query()
	selected, err := observationSelection(query, allMetrics)
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
	current, err := a.numeric(r.Context(), selected, time.Time{}, time.Time{}, "", l1TimeBasis)
	if err != nil {
		writeError(w, err)
		return
	}
	data := CurrentObservations{}
	for _, metric := range selected {
		data.setSample(metric, current[metric])
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

func historyQuery(query url.Values, now time.Time, nearEarthWind bool) (from, to time.Time, resolution string, limit int, bounded bool, err error) {
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
	if nearEarthWind {
		resolution = "1h"
	}
	if values := query[resolutionParameter]; len(values) > 0 {
		resolution = values[len(values)-1]
	}
	if !slices.Contains(observationResolutions, resolution) {
		err = invalid("resolution must be 1m, 5m, 1h or auto")
		return
	}
	if resolution == "auto" {
		switch {
		case nearEarthWind:
			resolution = "1h"
		case to.Sub(from) <= 6*time.Hour:
			resolution = "1m"
		case to.Sub(from) <= 24*time.Hour:
			resolution = "5m"
		default:
			resolution = "1h"
		}
	}
	if nearEarthWind && resolution != "1h" {
		err = invalid("Near-Earth wind history requires resolution=1h or auto")
		return
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

func (a *api) observationHistory(w http.ResponseWriter, r *http.Request, respond func(ObservationHistory)) {
	w.Header().Set("Cache-Control", "no-store")
	query := r.URL.Query()
	selected, err := observationSelection(query, historyDefaults)
	if err != nil {
		writeError(w, err)
		return
	}
	timeBasis := l1TimeBasis
	if values := query[timeBasisParameter]; len(values) > 0 {
		timeBasis = values[len(values)-1]
		if timeBasis != l1TimeBasis && timeBasis != nearEarthTimeBasis {
			writeError(w, invalid("time_basis must be l1 or near_earth"))
			return
		}
	}
	wind := selectedMetrics(selected, windMetrics)
	now := time.Now().UTC()
	from, to, resolution, limit, bounded, err := historyQuery(query, now, len(wind) > 0 && timeBasis == nearEarthTimeBasis)
	if err != nil {
		writeError(w, err)
		return
	}
	indices := selectedMetrics(selected, indexMetrics)
	hourly := selectedMetrics(selected, hourlyMetrics)
	xrs := selectedMetrics(selected, xrsMetrics)
	protons := selectedMetrics(selected, protonMetrics)
	days := map[string]int{"1m": 7, "5m": 31, "1h": 366}[resolution]
	if len(wind) > 0 && to.Sub(from) > time.Duration(days)*24*time.Hour {
		writeError(w, invalid("History interval must be at most "+strconv.Itoa(days)+" days for "+resolution))
		return
	}
	if len(indices) > 0 && to.Sub(from) > 31*24*time.Hour {
		writeError(w, invalid("Geomagnetic history interval must be at most 31 days"))
		return
	}
	if (len(xrs) > 0 || len(protons) > 0) && to.Sub(from) > 7*24*time.Hour {
		writeError(w, invalid("GOES history interval must be at most 7 days"))
		return
	}
	data := ObservationHistory{From: from, To: to, Series: make(map[string]HistorySeries)}
	if len(wind) > 0 {
		data.TimeBasis = &timeBasis
		data.ResolutionSeconds = pointer(int(resolutionDuration(resolution).Seconds()))
	}
	hourlyFrom := from
	if !bounded {
		hourlyFrom = to.Add(-time.Duration(limit) * time.Hour)
	}
	if len(hourly) > 0 && to.Sub(hourlyFrom) > 366*24*time.Hour {
		writeError(w, invalid("Use an interval of at most 366 days"))
		return
	}
	readFrom := from
	if len(hourly) > 0 && hourlyFrom.Before(readFrom) {
		readFrom = hourlyFrom
	}
	series, err := a.numeric(r.Context(), selected, readFrom, to, resolution, timeBasis)
	if err != nil {
		writeError(w, err)
		return
	}
	if len(hourly) > 0 {
		data.Points = pointer(hourlyPoints(series, hourly, hourlyFrom, to, limit))
	}
	for _, metric := range selected {
		result := series[metric]
		step := time.Duration(result.ResolutionSeconds) * time.Second
		properties := SeriesContext{TimeBasis: result.TimeBasis, Aggregation: result.Aggregation,
			ResolutionSeconds: result.ResolutionSeconds}
		seriesFrom := from
		if slices.Contains(hourly, metric) {
			seriesFrom = hourlyFrom
		}
		if duration := indexDuration[metric]; duration > 0 {
			points := overlapping(result.Points, seriesFrom, to, duration)
			items := make([]IndexSample, 0, len(points))
			for _, point := range points {
				end := point.ObservedAt.Add(duration)
				status := "completed"
				if end.After(now) {
					status = "in_progress"
				}
				items = append(items, IndexSample{IntervalStart: point.ObservedAt, IntervalEnd: end,
					IntervalStatus: status, Value: point.Value, Quality: quality(point), Provenance: provenance(point)})
			}
			data.Series[metric] = HistorySeries{Index: &IndexHistorySeries{Points: items, SeriesContext: properties,
				Coverage: observationCoverage(points, seriesFrom, to, now, duration, overlappingCoverage), DataStatus: indexDataStatus(metric)}}
			continue
		}
		coverageBasis := sampledCoverage
		if result.Aggregation == "mean" {
			coverageBasis = aggregateCoverage
		}
		points := result.Points
		items := make([]SolarSample, 0, len(points))
		first := 0
		for first < len(points) && points[first].ObservedAt.Before(seriesFrom) {
			first++
		}
		points = points[first:]
		for _, point := range points {
			item := SolarSample{ObservedAt: point.ObservedAt, Value: point.Value, Quality: quality(point),
				Provenance: provenance(point), Min: point.Min, Max: point.Max,
				Count: point.Count, ExpectedCount: point.ExpectedCount}
			if point.Count != nil && point.ExpectedCount != nil && *point.ExpectedCount > 0 {
				item.CoveragePercent = pointer(100 * float64(*point.Count) / float64(*point.ExpectedCount))
				item.IntervalEnd = pointer(point.ObservedAt.Add(step))
			}
			items = append(items, item)
		}
		data.Series[metric] = HistorySeries{Solar: &SolarHistorySeries{Points: items, SeriesContext: properties,
			Coverage: observationCoverage(points, seriesFrom, to, now, step, coverageBasis)}}
	}
	respond(data)
}

func resolutionDuration(resolution string) time.Duration {
	return map[string]time.Duration{"1m": time.Minute, "5m": 5 * time.Minute, "1h": time.Hour}[resolution]
}

func hourlyPoints(series map[string]observationcatalog.Series, metrics []string, from, to time.Time, limit int) []map[string]any {
	rows := map[time.Time]map[string]any{}
	var times []time.Time
	for _, metric := range metrics {
		for _, point := range series[metric].Points {
			if point.ObservedAt.Before(from) || !point.ObservedAt.Before(to) {
				continue
			}
			row := rows[point.ObservedAt]
			if row == nil {
				row = map[string]any{"issue_time": point.ObservedAt}
				for _, selected := range metrics {
					row[selected] = nil
				}
				rows[point.ObservedAt] = row
				times = append(times, point.ObservedAt)
			}
			row[metric] = point.Value
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
		if point.ObservedAt.Before(to) && point.ObservedAt.Add(duration).After(from) {
			result = append(result, point)
		}
	}
	return result
}
