package main

import (
	"context"
	"encoding/json"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/a-kryvenko/argus/observationcatalog"
)

type sample = observationcatalog.Point

func (a *api) numeric(ctx context.Context, metrics []string, from, to time.Time, interval, timeBasis string) (map[string]observationcatalog.Series, error) {
	query := url.Values{"metrics": {strings.Join(metrics, ",")}}
	limit := int64(latestResponseLimit)
	if !from.IsZero() {
		query.Set("from", from.Format(time.RFC3339Nano))
		query.Set("to", to.Format(time.RFC3339Nano))
		query.Set("interval", interval)
		limit = historyResponseLimit
	}
	if len(selectedMetrics(metrics, windMetrics)) > 0 {
		if timeBasis == "" {
			timeBasis = l1TimeBasis
		}
		query.Set(timeBasisParameter, timeBasis)
	}
	raw, _, err := a.request(ctx, a.clio, "GET", "/internal/v1/observations/series", query, nil, 90*time.Second, limit)
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
	now := time.Now().UTC()
	for _, id := range metrics {
		spec, _ := observationcatalog.Find(id)
		series := result.Data[id]
		step, basis, aggregation := seriesFormat(spec, from, interval, timeBasis)
		if series.Points == nil || series.ResolutionSeconds != int(step.Seconds()) ||
			series.TimeBasis != basis || series.Aggregation != aggregation || (from.IsZero() && len(series.Points) > 1) {
			return nil, unavailable("Observation service returned an invalid series")
		}
		var previous time.Time
		points := series.Points[:0]
		end := now
		if !to.IsZero() && to.Before(end) {
			end = to
		}
		for _, point := range series.Points {
			point.ObservedAt = point.ObservedAt.UTC()
			if point.ObservedAt.IsZero() || !point.ObservedAt.After(previous) ||
				!validSample(point, spec, basis, aggregation, step, !from.IsZero()) {
				return nil, unavailable("Observation service returned an invalid series")
			}
			previous = point.ObservedAt
			if withinSeries(point.ObservedAt, from, end, step, spec.SourceInterval, aggregation == "mean") {
				points = append(points, point)
			}
		}
		series.Points = points
		result.Data[id] = series
	}
	return result.Data, nil
}

func seriesFormat(metric observationcatalog.Metric, from time.Time, interval, windBasis string) (time.Duration, string, string) {
	step := time.Duration(metric.NativeSeconds) * time.Second
	basis, aggregation := "measurement", "source"
	if metric.SourceInterval {
		basis = "source_interval_start"
	}
	if metric.Group == "wind" {
		basis = windBasis
		if !from.IsZero() {
			step = resolutionDuration(interval)
			if basis == l1TimeBasis && step > time.Minute {
				aggregation = "mean"
			}
		}
	}
	return step, basis, aggregation
}

func validSample(point sample, metric observationcatalog.Metric, basis, aggregation string, step time.Duration, history bool) bool {
	if point.Quality != "" && point.Quality != "missing" && point.Quality != "flagged" {
		return false
	}
	if point.Value != nil && (point.Quality != "" || (!metric.Signed && *point.Value < 0) ||
		(metric.Max != nil && *point.Value > *metric.Max)) {
		return false
	}
	knownProduct := slices.Contains(metric.Products, point.SourceProduct) ||
		(history && slices.Contains(metric.ArchiveProducts, point.SourceProduct))
	if basis == nearEarthTimeBasis {
		knownProduct = point.SourceProduct == "omni.hourly"
	}
	if !knownProduct {
		return false
	}
	if (metric.SourceInterval || basis == nearEarthTimeBasis) && !point.ObservedAt.Equal(point.ObservedAt.Truncate(step)) {
		return false
	}
	if aggregation == "mean" {
		return point.ObservedAt.Equal(point.ObservedAt.Truncate(step)) &&
			point.Count != nil && point.ExpectedCount != nil && *point.ExpectedCount == int(step/time.Minute) &&
			*point.Count >= 0 && *point.Count <= *point.ExpectedCount && (*point.Count > 0) == (point.Value != nil)
	}
	return point.Count == nil && point.ExpectedCount == nil
}

func withinSeries(at, from, to time.Time, step time.Duration, sourceInterval, mean bool) bool {
	if from.IsZero() {
		return !at.After(to)
	}
	if !from.Before(to) || !at.Before(to) {
		return false
	}
	if sourceInterval {
		return at.Add(step).After(from)
	}
	return !at.Before(from) && (!mean || !at.Add(step).After(to))
}
