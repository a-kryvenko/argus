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
)

func observationCoverage(points []sample, start, end, now time.Time, step time.Duration, basis string) Coverage {
	if now.Before(end) {
		end = now
	}
	first := start.Truncate(step)
	if basis != overlappingCoverage && first.Before(start) {
		first = first.Add(step)
	}
	states := map[time.Time]string{}
	for _, point := range points {
		at := point.ObservedAt.Truncate(step)
		if point.Value != nil {
			states[at] = "usable"
		} else if point.Quality == "flagged" && states[at] != "usable" {
			states[at] = "invalid"
		}
	}
	gaps := []Gap{}
	var expected, usable, missing, invalid int
	for at := first; end.After(start) && at.Before(end); at = at.Add(step) {
		if basis == aggregateCoverage && at.Add(step).After(end) {
			break
		}
		expected++
		reason := states[at]
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
	wind, indices, hourly := selectedMetrics(selected, windMetrics), selectedMetrics(selected, indexMetrics), selectedMetrics(selected, hourlyMetrics)
	if metrics := slices.Concat(wind, indices); len(metrics) > 0 {
		past, err := a.numeric(ctx, metrics, now.Add(-7*time.Hour), now, "1m", "")
		if err != nil {
			return nil, err
		}
		for _, metric := range wind {
			changes[metric] = windChange(past[metric].Points, now, a.freshness[metric])
		}
		for _, metric := range indices {
			changes[metric] = nil
			latest := current[metric].Points
			points := past[metric].Points
			if len(latest) == 0 || len(points) == 0 || latest[len(latest)-1].Value == nil ||
				now.Sub(latest[len(latest)-1].ObservedAt.Add(indexDuration[metric])) > a.freshness[metric] {
				continue
			}
			point := points[len(points)-1]
			for i := len(points) - 1; i >= 0; i-- {
				if !points[i].ObservedAt.After(now.Add(-time.Hour)) {
					changes[metric] = valueChange(point, points[i])
					break
				}
			}
		}
	}
	if len(hourly) > 0 {
		past, err := a.numeric(ctx, hourly, now.Add(-48*time.Hour), now.Add(-time.Hour+time.Microsecond), "1m", "")
		if err != nil {
			return nil, err
		}
		for _, metric := range hourly {
			changes[metric] = nil
			if len(current[metric].Points) == 0 {
				continue
			}
			rows := past[metric].Points
			for i := len(rows) - 1; i >= 0; i-- {
				if rows[i].Value != nil {
					changes[metric] = valueChange(current[metric].Points[len(current[metric].Points)-1], rows[i])
					break
				}
			}
		}
	}
	return changes, nil
}

func valueChange(current, previous sample) *float64 {
	if current.Value == nil || previous.Value == nil {
		return nil
	}
	return pointer(roundChange(*current.Value - *previous.Value))
}

func windChange(points []sample, now time.Time, freshness time.Duration) *float64 {
	if len(points) == 0 {
		return nil
	}
	last := points[len(points)-1]
	if last.Value == nil || now.Sub(last.ObservedAt) > freshness {
		return nil
	}
	var recent, previous float64
	var recentCount, previousCount int
	for _, point := range points {
		if point.Value == nil {
			continue
		}
		age := last.ObservedAt.Sub(point.ObservedAt)
		switch {
		case age >= 0 && age < 5*time.Minute:
			recent += *point.Value
			recentCount++
		case age >= time.Hour && age < 65*time.Minute:
			previous += *point.Value
			previousCount++
		}
	}
	if recentCount == 0 || previousCount == 0 {
		return nil
	}
	return pointer(roundChange(recent/float64(recentCount) - previous/float64(previousCount)))
}

func roundChange(value float64) float64 {
	rounded, _ := strconv.ParseFloat(strconv.FormatFloat(value, 'f', 3, 64), 64)
	return rounded
}
