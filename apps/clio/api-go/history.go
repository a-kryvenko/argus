package main

import (
	"context"
	"errors"
	"math"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/a-kryvenko/argus/observationcatalog"
	"github.com/jackc/pgx/v5"
)

const maxHistoryRows = 250_000

const (
	l1TimeBasis        = "l1"
	nearEarthTimeBasis = "near_earth"
)

var historyOverflow = errors.New("Numeric read exceeds the supported batch size")

var normalizedMetrics = []string{"bx", "by", "bz", "v", "n", "t", "kp", "dst", "ap", "f10_7", "s10", "m10", "y10"}
var windMetrics = observationcatalog.IDs("wind")
var fieldComponents = []string{"bx", "by", "bz"}
var xrsMetrics = observationcatalog.IDs("xrs")
var protonMetrics = observationcatalog.IDs("protons")

type numericRow struct {
	values        map[string]*float64
	windTimeBasis string
}

type numericPoints map[time.Time]*numericRow

func (points numericPoints) set(at time.Time, metric string, value *float64) {
	at = at.UTC()
	if points[at] == nil {
		points[at] = &numericRow{values: make(map[string]*float64)}
	}
	points[at].values[metric] = value
}

func (points numericPoints) jsonPoints(selected []string, modelInputs bool) []map[string]any {
	times := make([]time.Time, 0, len(points))
	for at := range points {
		times = append(times, at)
	}
	slices.SortFunc(times, time.Time.Compare)
	output := make([]map[string]any, 0, len(times))
	for _, at := range times {
		point := points[at]
		row := map[string]any{"time": apiTime{at}}
		if selected == nil {
			for metric, value := range point.values {
				row[metric] = value
			}
		} else {
			for _, metric := range selected {
				if value, found := point.values[metric]; found {
					row[metric] = value
				}
			}
		}
		if modelInputs && point.windTimeBasis != "" {
			row["wind_time_basis"] = point.windTimeBasis
		}
		output = append(output, row)
	}
	return output
}

func (api api) history(w http.ResponseWriter, r *http.Request) {
	api.readHistory(w, r, false)
}

func (api api) modelInputs(w http.ResponseWriter, r *http.Request) {
	api.readHistory(w, r, true)
}

func (api api) readHistory(w http.ResponseWriter, r *http.Request, modelInputs bool) {
	query := requestQuery(r)
	selected := selectedMetrics(w, query)
	if selected == nil {
		return
	}
	validation := queryValidation{values: query}
	start := validation.dateTime("from")
	end := validation.dateTime("to")
	var timeBasis string
	if values := query["time_basis"]; len(values) != 0 {
		timeBasis = values[len(values)-1]
		if timeBasis != l1TimeBasis && timeBasis != nearEarthTimeBasis {
			const expected = "'l1' or 'near_earth'"
			validation.issues = append(validation.issues, queryIssue{
				Type: "literal_error", Loc: [2]string{"query", "time_basis"}, Input: &timeBasis,
				Message: "Input should be " + expected, Context: map[string]string{"expected": expected},
			})
		}
	}
	interval := "1h"
	if modelInputs && (query.Has("time_basis") || query.Has("interval")) {
		writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Model inputs use the prepared hourly series")
		return
	}
	if values := query["interval"]; len(values) != 0 {
		interval = values[len(values)-1]
		if interval != "1m" && interval != "5m" && interval != "1h" {
			const expected = "'1m', '5m' or '1h'"
			validation.issues = append(validation.issues, queryIssue{
				Type: "literal_error", Loc: [2]string{"query", "interval"}, Input: &interval,
				Message: "Input should be " + expected, Context: map[string]string{"expected": expected},
			})
		}
	}
	if !validation.valid(w) {
		return
	}
	if !validRange(start, end) {
		writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Use an interval of at most 366 days")
		return
	}
	if !query.Has("metrics") {
		switch {
		case timeBasis != "":
			selected = windMetrics
		case interval == "1m":
			selected = slices.Concat(legacyMetrics, xrsMetrics)
		case interval == "5m":
			selected = slices.Concat(legacyMetrics, protonMetrics)
		default:
			selected = legacyMetrics
		}
	}
	if reason := unsupportedHistory(selected, interval, timeBasis); reason != "" {
		writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", reason)
		return
	}
	if timeBasis == l1TimeBasis && interval != "1m" {
		if now := time.Now().UTC(); now.Before(end) {
			end = now
		}
	}
	points, err := api.db.history(r.Context(), selected, start, end, interval, timeBasis)
	if err != nil {
		writeStorageError(w, "Numeric observation storage is unavailable", err, historyOverflow)
		return
	}
	data := map[string]any{"interval": interval, "points": points.jsonPoints(selected, modelInputs)}
	if timeBasis != "" {
		data["time_basis"] = timeBasis
	}
	writeSuccess(w, data)
}

func unsupportedHistory(selected []string, interval, timeBasis string) string {
	if timeBasis == nearEarthTimeBasis && interval != "1h" {
		return "Near-Earth wind history requires an interval of 1h"
	}
	for _, metric := range selected {
		switch {
		case timeBasis != "" && !slices.Contains(windMetrics, metric):
			return "time_basis is supported for solar-wind metrics only"
		case interval != "1m" && slices.Contains(xrsMetrics, metric):
			return "GOES X-ray history requires an interval of 1m"
		case interval != "5m" && slices.Contains(protonMetrics, metric):
			return "GOES proton history requires an interval of 5m"
		}
	}
	return ""
}

func (db postgres) history(ctx context.Context, selected []string, start, end time.Time, interval, timeBasis string) (numericPoints, error) {
	points := make(numericPoints)
	native := selected
	if interval == "1h" && timeBasis == "" {
		// Preserve the compact history of the original internal contract.
		if err := db.readHourly(ctx, points, selected, start, end); err != nil {
			return nil, err
		}
		native = nil
		for _, metric := range selected {
			if slices.Contains(observationcatalog.IDs("model"), metric) {
				native = append(native, metric)
			}
		}
	}
	if len(native) == 0 {
		return points, nil
	}
	step, _ := time.ParseDuration(interval)
	if timeBasis == "" {
		timeBasis = l1TimeBasis
	}
	series, err := db.series(ctx, seriesQuery{metrics: native, from: start, to: end, windStep: step, windBasis: timeBasis})
	if err != nil {
		return nil, err
	}
	for metric, series := range series {
		for _, point := range series.Points {
			if !point.ObservedAt.Before(start) {
				points.set(point.ObservedAt, metric, point.Value)
			}
		}
	}
	return points, nil
}

func withFieldComponents(metrics []string) []string {
	metrics = slices.Clone(metrics)
	for _, component := range fieldComponents {
		if !slices.Contains(metrics, component) {
			metrics = append(metrics, component)
		}
	}
	return metrics
}

func totalField(row map[string]*float64) *float64 {
	x, y, z := row["bx"], row["by"], row["bz"]
	if x == nil || y == nil || z == nil {
		return nil
	}
	bx, by, bz := *x, *y, *z
	value := math.Sqrt(bx*bx + by*by + bz*bz)
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return nil
	}
	return &value
}

func (db postgres) readHourly(ctx context.Context, points numericPoints, selected []string, start, end time.Time) error {
	var columns []string
	for _, metric := range selected {
		if slices.Contains(normalizedMetrics, metric) {
			columns = append(columns, metric)
		}
	}
	deriveTotalField := slices.Contains(selected, "bt")
	if deriveTotalField {
		columns = withFieldComponents(columns)
	}
	if len(columns) == 0 {
		return nil
	}
	identifiers := make([]string, 0, len(columns))
	for _, column := range columns {
		identifiers = append(identifiers, pgx.Identifier{column}.Sanitize())
	}
	query := "SELECT observed_at, wind_time_basis, " + strings.Join(identifiers, ", ") + ` FROM clio.normalized_observation
		WHERE observed_at >= $1 AND observed_at < $2 ORDER BY observed_at LIMIT $3`
	rows, err := db.pool.Query(ctx, query, start, end, maxHistoryRows+1)
	if err != nil {
		return err
	}
	defer rows.Close()
	for count := 0; rows.Next(); count++ {
		if count == maxHistoryRows {
			return historyOverflow
		}
		var at time.Time
		var basis *string
		values := make([]*float64, len(columns))
		targets := []any{&at, &basis}
		for i := range values {
			targets = append(targets, &values[i])
		}
		if err := rows.Scan(targets...); err != nil {
			return err
		}
		for i, metric := range columns {
			points.set(at, metric, validatedValue(metric, values[i]))
		}
		row := points[at.UTC()]
		row.windTimeBasis = "legacy"
		if basis != nil {
			row.windTimeBasis = *basis
		}
		if deriveTotalField {
			points.set(at, "bt", totalField(row.values))
		}
	}
	return rows.Err()
}

func minuteAverage(samples []*float64) *float64 {
	count := 0
	sum, correction := 0.0, 0.0
	for _, sample := range samples {
		if sample == nil {
			continue
		}
		count++
		value := *sample
		next := sum + value
		// Match Python's compensated sum before integer conversion.
		if math.Abs(sum) >= math.Abs(value) {
			correction += (sum - next) + value
		} else {
			correction += (value - next) + sum
		}
		sum = next
	}
	if count == 0 {
		return nil
	}
	if !math.IsNaN(correction) && !math.IsInf(correction, 0) {
		sum += correction
	}
	value := sum / float64(count)
	return &value
}
