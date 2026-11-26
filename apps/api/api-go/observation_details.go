package main

import (
	"context"
	"math"
	"slices"
	"strconv"
	"time"

	"github.com/a-kryvenko/argus/observationcatalog"
)

func indexDataStatus(metric string) string {
	if metric == "kp" || metric == "ap" {
		return "estimated"
	}
	return "realtime"
}

const (
	sampledCoverage     = "sample_timestamps"
	aggregateCoverage   = "available_aggregate_windows"
	overlappingCoverage = "overlapping_intervals"
	dailyCoverage       = "utc_days"
)

func observationCoverage(points []sample, start, end, now time.Time, step time.Duration, basis string) Coverage {
	if now.Before(end) {
		end = now
	}
	first := start.Truncate(step)
	if basis != overlappingCoverage && basis != dailyCoverage && first.Before(start) {
		first = first.Add(step)
	}
	states := map[time.Time]string{}
	for _, point := range points {
		at := point.ObservedAt.UTC().Truncate(step)
		if point.Value != nil {
			states[at] = "usable"
		} else if point.Quality == "flagged" && states[at] != "usable" {
			states[at] = "invalid"
		} else if states[at] == "" {
			states[at] = "missing"
		}
	}
	gaps := []Gap{}
	var expected, usable, missing, invalid int
	for at := first; end.After(start) && at.Before(end); at = at.Add(step) {
		if basis == aggregateCoverage && at.Add(step).After(end) {
			break
		}
		reason := states[at]
		// A daily publication can have any timestamp in its UTC calendar day.
		// Only complete days can be declared missing on cadence alone.
		if basis == dailyCoverage && reason == "" && (at.Before(start) || at.Add(step).After(end)) {
			continue
		}
		expected++
		switch reason {
		case "usable":
			usable++
			continue
		case "invalid":
			invalid++
		default:
			reason = "missing"
			missing++
		}
		left, right := at, at.Add(step)
		if left.Before(start) {
			left = start
		}
		if right.After(end) {
			right = end
		}
		if i := len(gaps) - 1; i >= 0 && gaps[i].To.Equal(left) && gaps[i].Reason == reason {
			gaps[i].To = right
			gaps[i].Slots++
		} else {
			gaps = append(gaps, Gap{left, right, reason, 1})
		}
	}
	var percent *float64
	if expected > 0 {
		percent = pointer(math.RoundToEven(10000*float64(usable)/float64(expected)) / 100)
	}
	return Coverage{ExpectedSlots: expected, UsableSlots: usable, MissingSlots: missing,
		InvalidSlots: invalid, Percent: percent, ResolutionSeconds: int(step.Seconds()),
		EvaluatedTo: end, Basis: basis, Gaps: gaps}
}

func (a *api) observationChanges(ctx context.Context, current map[string]observationcatalog.Series, selected []string, now time.Time) (map[string]*float64, error) {
	changes := make(map[string]*float64)
	wind := selectedMetrics(selected, windMetrics)
	var indices, daily []string
	for _, metric := range selected {
		if indexDuration[metric] > 0 {
			indices = append(indices, metric)
		} else if slices.Contains(hourlyMetrics, metric) {
			daily = append(daily, metric)
		}
	}
	if metrics := slices.Concat(wind, indices); len(metrics) > 0 {
		past, err := a.numeric(ctx, metrics, now.Add(-7*time.Hour), now, "1m", "")
		if err != nil {
			return nil, err
		}
		for _, metric := range wind {
			changes[metric] = nil
			if latest := current[metric].Points; len(latest) > 0 {
				changes[metric] = windChange(latest[len(latest)-1], past[metric].Points, now, a.freshness[metric])
			}
		}
		for _, metric := range indices {
			changes[metric] = nil
			latest := current[metric].Points
			if len(latest) > 0 {
				changes[metric] = intervalChange(latest[len(latest)-1], past[metric].Points, now,
					indexDuration[metric], a.freshness[metric])
			}
		}
	}
	if len(daily) > 0 {
		past, err := a.numeric(ctx, daily, now.Add(-48*time.Hour), now.Add(-time.Hour+time.Microsecond), "1m", "")
		if err != nil {
			return nil, err
		}
		for _, metric := range daily {
			changes[metric] = nil
			latest, rows := current[metric].Points, past[metric].Points
			if len(latest) == 0 || len(rows) == 0 || now.Sub(latest[len(latest)-1].ObservedAt) > a.freshness[metric] {
				continue
			}
			changes[metric] = valueChange(latest[len(latest)-1], rows[len(rows)-1])
		}
	}
	return changes, nil
}

func valueChange(current, previous sample) *float64 {
	if current.Value == nil || previous.Value == nil {
		return nil
	}
	return roundChange(*current.Value - *previous.Value)
}

func intervalChange(current sample, points []sample, now time.Time, duration, freshness time.Duration) *float64 {
	if current.Value == nil || now.Sub(current.ObservedAt.Add(duration)) > freshness {
		return nil
	}
	reference := now.Add(-time.Hour)
	if !current.ObservedAt.After(reference) && current.ObservedAt.Add(duration).After(reference) {
		return pointer(0.0)
	}
	for i := len(points) - 1; i >= 0; i-- {
		point := points[i]
		if !point.ObservedAt.After(reference) {
			if point.ObservedAt.Add(duration).After(reference) {
				return valueChange(current, point)
			}
			break
		}
	}
	return nil
}

func windChange(last sample, points []sample, now time.Time, freshness time.Duration) *float64 {
	if last.Value == nil || now.Sub(last.ObservedAt) > freshness {
		return nil
	}
	recent, recentCount := *last.Value, 1
	var previous float64
	var previousCount int
	for _, point := range points {
		if point.Value == nil {
			continue
		}
		age := last.ObservedAt.Sub(point.ObservedAt)
		switch {
		case age > 0 && age < 5*time.Minute:
			recent += *point.Value
			recentCount++
		case age >= time.Hour && age < 65*time.Minute:
			previous += *point.Value
			previousCount++
		}
	}
	if previousCount == 0 {
		return nil
	}
	return roundChange(recent/float64(recentCount) - previous/float64(previousCount))
}

func roundChange(value float64) *float64 {
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return nil
	}
	rounded, _ := strconv.ParseFloat(strconv.FormatFloat(value, 'f', 3, 64), 64)
	return &rounded
}
