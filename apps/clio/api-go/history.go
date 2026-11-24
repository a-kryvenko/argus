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

var historyOverflow = errors.New("Numeric read exceeds the supported batch size")

var normalizedMetrics = []string{"bx", "by", "bz", "v", "n", "t", "kp", "dst", "ap", "f10_7", "s10", "m10", "y10"}

// Numeric products collected in live mode by Clio.
var liveProducts = []string{
	"swpc.rtsw_plasma", "swpc.rtsw_mag", "swpc.kp", "swpc.dst", "swpc.f107", "celestrak.f107", "solfsmy.means",
}

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
	points, err := api.db.history(r.Context(), selected, start, end, interval)
	if err != nil {
		writeStorageError(w, "Numeric observation storage is unavailable", err, historyOverflow)
		return
	}
	writeSuccess(w, map[string]any{"interval": interval, "points": points.jsonPoints(selected)})
}

func (db postgres) history(ctx context.Context, selected []string, start, end time.Time, interval string) (numericPoints, error) {
	points := make(numericPoints)
	means := make([]string, 0)
	for _, metric := range selected {
		if strings.HasSuffix(metric, "_81mean") {
			means = append(means, metric)
		}
	}
	native := selected
	if interval == "1h" {
		if err := db.readHourly(ctx, points, selected, start, end); err != nil {
			return nil, err
		}
		native = means
	}
	if len(native) == 0 {
		return points, nil
	}
	const statement = `SELECT metric, observed_at, value FROM clio.measurement
		WHERE observed_at >= $1 AND observed_at < $2 AND metric = ANY($3::varchar[])
		  AND ($4::boolean OR source_product = ANY($5::varchar[]) OR metric = ANY($6::varchar[]))
		ORDER BY observed_at LIMIT $7`
	rows, err := db.pool.Query(ctx, statement, start, end, native, interval == "1h", liveProducts, means, maxHistoryRows+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	buckets := make(map[historyBucket][5]*float64)
	for count := 0; rows.Next(); count++ {
		if count == maxHistoryRows {
			return nil, historyOverflow
		}
		row, err := scanMeasurement(rows)
		if err != nil {
			return nil, err
		}
		row.value = validatedValue(row.metric, row.value)
		if interval != "5m" {
			points.set(row.observedAt, row.metric, row.value)
			continue
		}
		window := row.observedAt.UTC().Truncate(5 * time.Minute)
		if window.Before(start) || window.Add(5*time.Minute).After(end) {
			continue
		}
		key := historyBucket{window, row.metric}
		samples := buckets[key]
		samples[row.observedAt.UTC().Minute()%5] = row.value
		buckets[key] = samples
	}
	for key, samples := range buckets {
		points.set(key.time, key.metric, minuteAverage(samples))
	}
	return points, rows.Err()
}

func (db postgres) readHourly(ctx context.Context, points numericPoints, selected []string, start, end time.Time) error {
	var columns []string
	for _, metric := range selected {
		if slices.Contains(normalizedMetrics, metric) {
			columns = append(columns, metric)
		}
	}
	totalField := slices.Contains(selected, "bt")
	if totalField {
		for _, component := range []string{"bx", "by", "bz"} {
			if !slices.Contains(columns, component) {
				columns = append(columns, component)
			}
		}
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
		if totalField {
			row := points[at.UTC()]
			x, y, z := row["bx"], row["by"], row["bz"]
			var field *float64
			if x != nil && y != nil && z != nil {
				bx, by, bz := *x, *y, *z
				value := math.Sqrt(bx*bx + by*by + bz*bz)
				field = &value
			}
			points.set(at, "bt", field)
		}
	}
	return rows.Err()
}

type historyBucket struct {
	time   time.Time
	metric string
}

func minuteAverage(samples [5]*float64) *float64 {
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
