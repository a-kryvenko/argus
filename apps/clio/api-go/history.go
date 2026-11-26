package main

import (
	"context"
	"errors"
	"math"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

const maxHistoryRows = 250_000

const (
	l1TimeBasis        = "l1"
	nearEarthTimeBasis = "near_earth"
)

var historyOverflow = errors.New("Numeric read exceeds the supported batch size")

var normalizedMetrics = []string{"bx", "by", "bz", "v", "n", "t", "kp", "dst", "ap", "f10_7", "s10", "m10", "y10"}
var windMetrics = []string{"bx", "by", "bz", "bt", "v", "n", "t"}
var fieldComponents = []string{"bx", "by", "bz"}
var nativeWindProducts = []string{"swpc.rtsw_plasma", "swpc.rtsw_mag"}
var xrsMetrics = []string{"xrs_a", "xrs_b"}
var protonMetrics = []string{"proton_10mev", "proton_100mev"}

// Numeric products collected in live mode by Clio.
var liveProducts = slices.Concat(nativeWindProducts, []string{
	"swpc.kp", "swpc.dst", "swpc.f107", "celestrak.f107", "solfsmy.means", "goes.xrs", "goes.protons",
})

type numericPoints map[time.Time]map[string]*float64

func (points numericPoints) set(at time.Time, metric string, value *float64) {
	at = at.UTC()
	if points[at] == nil {
		points[at] = make(map[string]*float64)
	}
	points[at][metric] = value
}

func (points numericPoints) jsonPoints(selected []string) []map[string]any {
	times := make([]time.Time, 0, len(points))
	for at := range points {
		times = append(times, at)
	}
	slices.SortFunc(times, time.Time.Compare)
	output := make([]map[string]any, 0, len(times))
	for _, at := range times {
		values := points[at]
		row := map[string]any{"time": apiTime{at}}
		if selected == nil {
			for metric, value := range values {
				row[metric] = observationValue(metric, value)
			}
		} else {
			for _, metric := range selected {
				row[metric] = observationValue(metric, values[metric])
			}
		}
		output = append(output, row)
	}
	return output
}

func (api api) history(w http.ResponseWriter, r *http.Request) {
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
	data := map[string]any{"interval": interval, "points": points.jsonPoints(selected)}
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
	means := make([]string, 0)
	for _, metric := range selected {
		if strings.HasSuffix(metric, "_81mean") {
			means = append(means, metric)
		}
	}
	native := selected
	legacyHourly := interval == "1h" && timeBasis == ""
	if legacyHourly {
		if err := db.readHourly(ctx, points, selected, start, end); err != nil {
			return nil, err
		}
		native = means
	}
	if len(native) == 0 {
		return points, nil
	}
	products := liveProducts
	deriveTotalField := false
	switch timeBasis {
	case l1TimeBasis:
		products = nativeWindProducts
	case nearEarthTimeBasis:
		products = []string{"omni.hourly"}
		if slices.Contains(native, "bt") {
			native = withFieldComponents(native)
			deriveTotalField = true
		}
	}
	var bucketSize time.Duration
	if interval == "5m" {
		bucketSize = 5 * time.Minute
	} else if interval == "1h" && timeBasis == l1TimeBasis {
		bucketSize = time.Hour
	}
	const statement = `SELECT metric, observed_at, value FROM clio.measurement
		WHERE observed_at >= $1 AND observed_at < $2 AND metric = ANY($3::varchar[])
		  AND ($4::boolean OR source_product = ANY($5::varchar[]) OR metric = ANY($6::varchar[]))
		ORDER BY observed_at LIMIT $7`
	rows, err := db.pool.Query(ctx, statement, start, end, native, legacyHourly, products, means, maxHistoryRows+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	buckets := make(historyBuckets)
	for count := 0; rows.Next(); count++ {
		if count == maxHistoryRows {
			return nil, historyOverflow
		}
		row, err := scanMeasurement(rows)
		if err != nil {
			return nil, err
		}
		row.value = validatedValue(row.metric, row.value)
		if bucketSize == 0 || slices.Contains(protonMetrics, row.metric) {
			points.set(row.observedAt, row.metric, row.value)
			continue
		}
		buckets.add(row, start, end, bucketSize)
	}
	for key, samples := range buckets {
		points.set(key.time, key.metric, minuteAverage(samples))
	}
	if deriveTotalField {
		for at, row := range points {
			points.set(at, "bt", totalField(row))
		}
	}
	return points, rows.Err()
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
	query := "SELECT observed_at, " + strings.Join(identifiers, ", ") + ` FROM clio.normalized_observation
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
		values := make([]*float64, len(columns))
		targets := []any{&at}
		for i := range values {
			targets = append(targets, &values[i])
		}
		if err := rows.Scan(targets...); err != nil {
			return err
		}
		for i, metric := range columns {
			points.set(at, metric, validatedValue(metric, values[i]))
		}
		if deriveTotalField {
			points.set(at, "bt", totalField(points[at.UTC()]))
		}
	}
	return rows.Err()
}

type historyBucket struct {
	time   time.Time
	metric string
}

type historyBuckets map[historyBucket][]*float64

func (buckets historyBuckets) add(row measurement, start, end time.Time, step time.Duration) {
	window := row.observedAt.UTC().Truncate(step)
	if window.Before(start) || window.Add(step).After(end) {
		return
	}
	key := historyBucket{window, row.metric}
	if buckets[key] == nil {
		buckets[key] = make([]*float64, int(step/time.Minute))
	}
	buckets[key][int(row.observedAt.Sub(window)/time.Minute)] = row.value
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
