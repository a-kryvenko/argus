package main

import (
	"context"
	"math"
	"slices"
	"strconv"
	"time"
)

func observationMetadata(metric, interval, timeBasis string) SeriesMetadata {
	const base = "https://services.swpc.noaa.gov"
	switch metric {
	case "xrs_a", "xrs_b":
		label := "XRS-A solar X-ray flux (0.5–4 Å)"
		if metric == "xrs_b" {
			label = "XRS-B solar X-ray flux (1–8 Å)"
		}
		return SeriesMetadata{Label: label, Unit: "W/m²", Source: "NOAA SWPC GOES XRS",
			SourceURL: base + "/json/goes/primary/xrays-7-day.json", TimeBasis: "measurement",
			ResolutionSeconds: 60, Aggregation: pointer("source_1_minute"), GapFilling: pointer("none"),
			Selection: pointer("NOAA primary GOES"), PollSeconds: pointer(60)}
	case "proton_10mev", "proton_100mev":
		label := "Integral proton flux ≥10 MeV"
		if metric == "proton_100mev" {
			label = "Integral proton flux ≥100 MeV"
		}
		return SeriesMetadata{Label: label, Unit: "pfu", Source: "NOAA SWPC GOES",
			SourceURL: base + "/json/goes/primary/integral-protons-7-day.json", TimeBasis: "measurement",
			ResolutionSeconds: 300, Aggregation: pointer("source_5_minute"), GapFilling: pointer("none"),
			Selection: pointer("NOAA primary GOES"), PollSeconds: pointer(300)}
	}
	label := map[string]string{"bx": "Bx", "by": "By", "bz": "Bz", "bt": "Total magnetic field",
		"v": "Solar wind speed", "n": "Proton density", "t": "Proton temperature",
		"kp": "Estimated Kp", "dst": "Real-time Dst"}[metric]
	unit := "nT"
	if known, ok := map[string]string{"v": "km/s", "n": "cm⁻³", "t": "K", "kp": ""}[metric]; ok {
		unit = known
	}
	if slices.Contains(indexMetrics, metric) {
		source, path, poll := "NOAA SWPC", "/products/noaa-planetary-k-index.json", 60
		if metric == "dst" {
			source, path, poll = "WDC Kyoto via NOAA SWPC", "/products/kyoto-dst.json", 300
		}
		return SeriesMetadata{Label: label, Unit: unit, Source: source, SourceURL: base + path,
			TimeBasis: "source_interval_start", ResolutionSeconds: int(indexDuration[metric].Seconds()),
			FreshnessBasis: pointer("interval_end"), GapFilling: pointer("none"), PollSeconds: &poll}
	}
	path := "/json/rtsw/rtsw_wind_1m.json"
	if slices.Contains([]string{"bx", "by", "bz", "bt"}, metric) {
		path = "/json/rtsw/rtsw_mag_1m.json"
	}
	result := SeriesMetadata{Label: label, Unit: unit, ResolutionSeconds: int(resolutionDuration(interval).Seconds()),
		GapFilling: pointer("none")}
	if timeBasis == nearEarthTimeBasis {
		result.Source = "NASA OMNI2"
		result.SourceURL = "https://omniweb.gsfc.nasa.gov/omniweb/html/ow_data.html"
		result.TimeBasis = "near_earth_hour_start"
		result.Location = pointer("near_earth")
		result.Propagated = pointer(true)
		result.Aggregation = pointer("source_hourly_mean")
		if metric == "bt" {
			result.Label = "Magnitude of hourly mean magnetic field vector"
			result.Aggregation = pointer("norm_of_hourly_mean_components")
		}
	} else {
		result.Source = "NOAA SWPC RTSW"
		result.SourceURL = base + path
		result.TimeBasis = "measurement"
		result.Location = pointer("L1")
		result.Propagated = pointer(false)
		result.Selection = pointer("NOAA active spacecraft")
		aggregation := "source_1_minute"
		if interval != "1m" {
			aggregation = "mean"
		}
		result.Aggregation = &aggregation
	}
	if slices.Contains([]string{"bx", "by", "bz"}, metric) {
		result.CoordinateSystem = pointer("GSM")
	}
	return result
}

func indexDataStatus(metric string) string {
	if metric == "kp" {
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
		at := point.at.Truncate(step)
		if point.value != nil {
			states[at] = "usable"
		} else if states[at] != "usable" {
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

func (a *api) observationChanges(ctx context.Context, current map[string][]sample, selected []string, now time.Time) (map[string]*float64, error) {
	changes := make(map[string]*float64)
	wind, indices, hourly := selectedMetrics(selected, windMetrics), selectedMetrics(selected, indexMetrics), selectedMetrics(selected, hourlyMetrics)
	if metrics := slices.Concat(wind, indices); len(metrics) > 0 {
		past, err := a.numeric(ctx, "history", metrics, now.Add(-7*time.Hour), now, "1m", "")
		if err != nil {
			return nil, err
		}
		for _, metric := range wind {
			changes[metric] = windChange(past[metric], now, a.freshness[metric])
		}
		for _, metric := range indices {
			changes[metric] = nil
			latest := current[metric]
			points := past[metric]
			if len(latest) == 0 || len(points) == 0 || latest[len(latest)-1].value == nil ||
				now.Sub(latest[len(latest)-1].at.Add(indexDuration[metric])) > a.freshness[metric] {
				continue
			}
			point := points[len(points)-1]
			for i := len(points) - 1; i >= 0; i-- {
				if !points[i].at.After(now.Add(-time.Hour)) {
					changes[metric] = valueChange(point, points[i])
					break
				}
			}
		}
	}
	if len(hourly) > 0 {
		past, err := a.numeric(ctx, "history", hourly, now.Add(-48*time.Hour), now.Add(-time.Hour+time.Microsecond), "1m", "")
		if err != nil {
			return nil, err
		}
		for _, metric := range hourly {
			changes[metric] = nil
			if len(current[metric]) == 0 {
				continue
			}
			rows := past[metric]
			for i := len(rows) - 1; i >= 0; i-- {
				if rows[i].value != nil {
					changes[metric] = valueChange(current[metric][len(current[metric])-1], rows[i])
					break
				}
			}
		}
	}
	return changes, nil
}

func valueChange(current, previous sample) *float64 {
	if current.value == nil || previous.value == nil {
		return nil
	}
	return pointer(roundChange(*current.value - *previous.value))
}

func windChange(points []sample, now time.Time, freshness time.Duration) *float64 {
	if len(points) == 0 {
		return nil
	}
	last := points[len(points)-1]
	if last.value == nil || now.Sub(last.at) > freshness {
		return nil
	}
	var recent, previous float64
	var recentCount, previousCount int
	for _, point := range points {
		if point.value == nil {
			continue
		}
		age := last.at.Sub(point.at)
		switch {
		case age >= 0 && age < 5*time.Minute:
			recent += *point.value
			recentCount++
		case age >= time.Hour && age < 65*time.Minute:
			previous += *point.value
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
