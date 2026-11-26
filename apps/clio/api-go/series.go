package main

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"time"

	"github.com/a-kryvenko/argus/observationcatalog"
	"github.com/jackc/pgx/v5"
)

type seriesQuery struct {
	metrics    []string
	from, to   time.Time
	windStep   time.Duration
	windBasis  string
	receivedBy *time.Time
}

func (api api) series(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	selected := selectedMetrics(w, query)
	if selected == nil {
		return
	}
	now := time.Now().UTC()
	read := seriesQuery{metrics: selected, to: now, windStep: time.Minute, windBasis: l1TimeBasis}
	if query.Has("from") || query.Has("to") {
		from, fromErr := time.Parse(time.RFC3339Nano, query.Get("from"))
		to, toErr := time.Parse(time.RFC3339Nano, query.Get("to"))
		if fromErr != nil || toErr != nil || !validRange(from, to) {
			writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Use RFC3339 from and to with a positive range of at most 366 days")
			return
		}
		read.from, read.to = from.UTC(), to.UTC()
		if now.Before(read.to) {
			read.to = now
		}
	}
	if query.Has("time_basis") {
		read.windBasis = query.Get("time_basis")
	}
	if query.Has("received_by") {
		cutoff, err := time.Parse(time.RFC3339Nano, query.Get("received_by"))
		if err != nil || read.from.IsZero() {
			writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Use RFC3339 received_by with a from/to range")
			return
		}
		cutoff = cutoff.UTC()
		if now.Before(cutoff) {
			cutoff = now
		}
		read.receivedBy = &cutoff
	}
	interval := query.Get("interval")
	switch interval {
	case "":
		if read.windBasis == nearEarthTimeBasis {
			read.windStep = time.Hour
		}
	case "1m":
	case "5m":
		read.windStep = 5 * time.Minute
	case "1h":
		read.windStep = time.Hour
	default:
		writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Use an interval of 1m, 5m or 1h")
		return
	}
	if read.windBasis != l1TimeBasis && read.windBasis != nearEarthTimeBasis ||
		read.windBasis == nearEarthTimeBasis && (read.from.IsZero() || read.windStep != time.Hour) {
		writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Use l1 or request a near_earth hourly range")
		return
	}
	result, err := api.db.series(r.Context(), read)
	if errors.Is(err, historyOverflow) {
		writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Observation range exceeds the supported batch size")
		return
	}
	if err != nil {
		writeStorageError(w, "Numeric observation storage is unavailable", err, nil)
		return
	}
	writeSuccess(w, result)
}

type readProduct struct {
	metric   observationcatalog.Metric
	products []string
}

func (db postgres) series(ctx context.Context, query seriesQuery) (map[string]observationcatalog.Series, error) {
	result := make(map[string]observationcatalog.Series, len(query.metrics))
	plan := make(map[string]readProduct, len(query.metrics))
	var products []string
	var latestMetrics, latestProducts []string
	left := query.from
	for _, id := range query.metrics {
		metric, found := observationcatalog.Find(id)
		if !found {
			continue
		}
		current := observationcatalog.Series{Points: []observationcatalog.Point{}, ResolutionSeconds: metric.NativeSeconds,
			TimeBasis: "measurement", Aggregation: "source"}
		allowed := db.readProducts(metric, !query.from.IsZero())
		if metric.Group == "wind" {
			current.TimeBasis = query.windBasis
			if !query.from.IsZero() {
				current.ResolutionSeconds = int(query.windStep.Seconds())
				if query.windBasis == nearEarthTimeBasis {
					allowed = []string{"omni.hourly"}
				} else if query.windStep != time.Minute {
					current.Aggregation = "mean"
				}
			}
		}
		if metric.SourceInterval {
			current.TimeBasis = "source_interval_start"
			if !query.from.IsZero() {
				start := query.from.Truncate(time.Duration(metric.NativeSeconds) * time.Second)
				if start.Before(left) {
					left = start
				}
			}
		}
		result[id] = current
		plan[id] = readProduct{metric, allowed}
		for _, product := range allowed {
			if !slices.Contains(products, product) {
				products = append(products, product)
			}
			if query.from.IsZero() {
				latestMetrics = append(latestMetrics, id)
				latestProducts = append(latestProducts, product)
			}
		}
	}
	if len(products) == 0 || !query.from.IsZero() && !query.to.After(query.from) {
		return result, nil
	}
	const fields = "metric, observed_at, value, source_product, received_at, quality, spacecraft"
	const historySQL = "SELECT " + fields + ` FROM clio.measurement
		WHERE metric = ANY($1::varchar[]) AND source_product = ANY($2::varchar[])
		  AND observed_at >= $3 AND observed_at < $4
		  AND ($6::timestamptz IS NULL OR received_at BETWEEN observed_at AND $6) ORDER BY observed_at LIMIT $5`
	const latestSQL = `SELECT sample.*
		FROM unnest($1::varchar[], $2::varchar[]) AS requested(metric, product)
		CROSS JOIN LATERAL (SELECT ` + fields + ` FROM clio.measurement
			WHERE metric = requested.metric AND source_product = requested.product
			  AND observed_at <= $3 ORDER BY observed_at DESC LIMIT 1) AS sample`
	var rows pgx.Rows
	var err error
	if query.from.IsZero() {
		rows, err = db.pool.Query(ctx, latestSQL, latestMetrics, latestProducts, query.to)
	} else {
		rows, err = db.pool.Query(ctx, historySQL, query.metrics, products, left, query.to, maxHistoryRows+1, query.receivedBy)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type sourceSlot struct {
		at      time.Time
		product string
	}
	pointsByMetric := make(map[string]map[sourceSlot]observationcatalog.Point)
	for count := 0; rows.Next(); count++ {
		if count >= maxHistoryRows {
			return nil, historyOverflow
		}
		var id string
		var point observationcatalog.Point
		var quality, spacecraft *string
		err := rows.Scan(&id, &point.ObservedAt, &point.Value, &point.SourceProduct, &point.ReceivedAt, &quality, &spacecraft)
		if err != nil {
			return nil, err
		}
		spec := plan[id]
		rank := slices.Index(spec.products, point.SourceProduct)
		if rank < 0 {
			continue
		}
		point.ObservedAt = point.ObservedAt.UTC()
		if quality != nil {
			point.Quality = *quality
		}
		if spacecraft != nil {
			point.Spacecraft = *spacecraft
		}
		point.Value = validatedValue(id, point.Value)
		if point.Quality != "" || point.Value == nil {
			point.Value = nil
			if point.Quality == "" {
				point.Quality = "missing"
			}
		}
		at := point.ObservedAt
		if !query.from.IsZero() && at.Before(query.from) && !spec.metric.SourceInterval {
			continue
		}
		if spec.metric.ID == "f10_7" {
			day := at.Truncate(24 * time.Hour)
			// The forecast input series uses the same Penticton noon flux as the daily target.
			if query.receivedBy != nil && point.SourceProduct == "swpc.f107" && !at.Equal(day.Add(20*time.Hour)) {
				continue
			}
			at = day
		}
		byTime := pointsByMetric[id]
		if byTime == nil {
			byTime = make(map[sourceSlot]observationcatalog.Point)
			pointsByMetric[id] = byTime
		}
		key := sourceSlot{at, point.SourceProduct}
		if old, exists := byTime[key]; !exists || point.ObservedAt.After(old.ObservedAt) {
			byTime[key] = point
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for id, series := range result {
		allowed := plan[id].products
		if len(allowed) <= 1 {
			for _, point := range pointsByMetric[id] {
				series.Points = append(series.Points, point)
			}
		} else {
			best := make(map[time.Time]observationcatalog.Point)
			for key, point := range pointsByMetric[id] {
				old, exists := best[key.at]
				if !exists || preferPoint(point, old, slices.Index(allowed, point.SourceProduct), slices.Index(allowed, old.SourceProduct)) {
					best[key.at] = point
				}
			}
			for _, point := range best {
				series.Points = append(series.Points, point)
			}
		}
		slices.SortFunc(series.Points, func(a, b observationcatalog.Point) int {
			return a.ObservedAt.Compare(b.ObservedAt)
		})
		if query.from.IsZero() && len(series.Points) > 1 {
			series.Points = series.Points[len(series.Points)-1:]
		} else if series.Aggregation == "mean" {
			series.Points = aggregateWind(series.Points, query.from, query.to, query.windStep)
		}
		result[id] = series
	}
	return result, nil
}

func (db postgres) readProducts(metric observationcatalog.Metric, history bool) []string {
	live, archive := metric.Products, metric.ArchiveProducts
	if policy, found := db.sources[metric.ID]; found {
		live = intersectProducts(policy.Sources.Live, live)
		archive = intersectProducts(policy.Sources.Historical, archive)
	}
	if !history {
		return live
	}
	return slices.Concat(live, archive)
}

func intersectProducts(priority, allowed []string) []string {
	var matches []string
	for _, product := range priority {
		if slices.Contains(allowed, product) {
			matches = append(matches, product)
		}
	}
	return matches
}

func preferPoint(point, old observationcatalog.Point, rank, oldRank int) bool {
	if rank == oldRank {
		return point.ObservedAt.After(old.ObservedAt)
	}
	if (point.Value == nil) != (old.Value == nil) {
		return point.Value != nil
	}
	return rank < oldRank
}

func aggregateWind(points []observationcatalog.Point, from, to time.Time, step time.Duration) []observationcatalog.Point {
	out := make([]observationcatalog.Point, 0)
	expected := int(step / time.Minute)
	for start := 0; start < len(points); {
		at := points[start].ObservedAt.Truncate(step)
		end := start + 1
		for end < len(points) && points[end].ObservedAt.Before(at.Add(step)) {
			end++
		}
		if !at.Before(from) && !at.Add(step).After(to) {
			byMinute := make([]observationcatalog.Point, expected)
			bucket := observationcatalog.Point{ObservedAt: at, SourceProduct: points[start].SourceProduct,
				ExpectedCount: &expected, Quality: "missing"}
			for _, sample := range points[start:end] {
				byMinute[int(sample.ObservedAt.Sub(at)/time.Minute)] = sample
			}
			count := 0
			values := make([]*float64, 0, expected)
			unknownReceipt := false
			for _, sample := range byMinute {
				value := sample.Value
				if sample.ReceivedAt != nil && (bucket.ReceivedAt == nil || sample.ReceivedAt.After(*bucket.ReceivedAt)) {
					bucket.ReceivedAt = sample.ReceivedAt
				}
				if value == nil {
					if sample.Quality == "flagged" {
						bucket.Quality = "flagged"
					}
					continue
				}
				values = append(values, value)
				if count == 0 {
					bucket.Spacecraft = sample.Spacecraft
				} else if sample.Spacecraft != bucket.Spacecraft {
					bucket.Spacecraft = ""
				}
				count++
				if sample.ReceivedAt == nil {
					unknownReceipt = true
				}
				if bucket.Min == nil || *value < *bucket.Min {
					bucket.Min = value
				}
				if bucket.Max == nil || *value > *bucket.Max {
					bucket.Max = value
				}
			}
			bucket.Value, bucket.Count = minuteAverage(values), &count
			if count > 0 {
				bucket.Quality = ""
			}
			if unknownReceipt {
				bucket.ReceivedAt = nil
			}
			out = append(out, bucket)
		}
		start = end
	}
	return out
}
