package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
	"github.com/jackc/pgx/v5"
)

const densityProduct = "atmospheric-density"
const eventsProduct = "solar-wind-events"
const projectionInterval = 15 * time.Second
const projectionReconnect = 2 * time.Second
const projectionChannel = "prophet_release_published"
const preparedForecastPath = "/internal/v1/public-forecasts/"

type projectionSource interface {
	latestReleaseIDs(context.Context) (map[string]string, error)
	readRelease(context.Context, string, string) (release, error)
}

type preparedForecast struct {
	releaseID         string
	issueTime         time.Time
	compact, withMeta []byte
}

type preparedForecasts struct {
	source projectionSource
	mutex  sync.RWMutex
	latest map[string]preparedForecast
	synced bool
}

func newPreparedForecasts(source projectionSource) *preparedForecasts {
	return &preparedForecasts{source: source, latest: make(map[string]preparedForecast)}
}

func (p *preparedForecasts) get(product string) (preparedForecast, bool) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	return p.latest[product], p.synced
}

func (p *preparedForecasts) set(product string, result preparedForecast) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.latest[product] = result
}

func (p *preparedForecasts) run(ctx context.Context, wake <-chan struct{}) {
	ticker := time.NewTicker(projectionInterval)
	defer ticker.Stop()
	for {
		budget, cancel := context.WithTimeout(ctx, 30*time.Second)
		p.refresh(budget)
		cancel()
		select {
		case <-ctx.Done():
			return
		case <-wake:
		case <-ticker.C:
		}
	}
}

func subscribeToPublications(ctx context.Context, dsn string, wake chan<- struct{}) {
	signal := func() {
		select {
		case wake <- struct{}{}:
		default:
		}
	}
	for ctx.Err() == nil {
		connect, cancel := context.WithTimeout(ctx, 5*time.Second)
		listener, err := pgx.Connect(connect, dsn)
		if err == nil {
			_, err = listener.Exec(connect, "LISTEN "+projectionChannel)
		}
		cancel()
		if err == nil {
			// Close the gap between the first refresh and the committed LISTEN.
			signal()
			for ctx.Err() == nil {
				if _, err = listener.WaitForNotification(ctx); err != nil {
					break
				}
				signal()
			}
		}
		if listener != nil {
			closeContext, cancel := context.WithTimeout(context.Background(), time.Second)
			listener.Close(closeContext)
			cancel()
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(projectionReconnect):
		}
	}
}

func (db postgres) latestReleaseIDs(ctx context.Context) (map[string]string, error) {
	rows, err := db.pool.Query(ctx, `SELECT product, id::text FROM (`+latestReleases+`) releases`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	latest := make(map[string]string)
	for rows.Next() {
		var product, id string
		if err := rows.Scan(&product, &id); err != nil {
			return nil, err
		}
		latest[product] = id
	}
	return latest, rows.Err()
}

func (p *preparedForecasts) refresh(ctx context.Context) {
	ids, err := p.source.latestReleaseIDs(ctx)
	p.mutex.Lock()
	p.synced = err == nil
	p.mutex.Unlock()
	if err != nil {
		if ctx.Err() == nil {
			log.Printf("read forecast projection sources: %v", err)
		}
		return
	}
	for _, item := range products {
		target := item.Name
		if !hasPreparedForecast(target) {
			continue
		}
		id := ids[target]
		previous, _ := p.get(target)
		if id == previous.releaseID {
			continue
		}
		if id == "" {
			p.set(target, preparedForecast{})
			continue
		}
		// Requests must not keep serving the previous release during preparation.
		p.set(target, preparedForecast{})
		source, err := p.source.readRelease(ctx, target, id)
		if err != nil {
			if ctx.Err() == nil {
				log.Printf("read public forecast %s release %s: %v", target, id, err)
			}
			continue
		}
		result, err := prepareForecast(source)
		if err != nil {
			log.Printf("prepare public forecast %s release %s: %v", target, id, err)
		}
		p.set(target, result)
	}
}

func hasPreparedForecast(target string) bool {
	_, known := publicforecast.Products[target]
	return known || target == densityProduct || target == eventsProduct
}

func prepareForecast(source release) (preparedForecast, error) {
	result := preparedForecast{releaseID: source.ID, issueTime: source.IssueTime.UTC()}
	frames := make(map[string][]publicforecast.Row, len(source.Artifacts))
	for _, item := range source.Artifacts {
		rows, err := forecastRows(item, source.IssueTime)
		if err != nil {
			return result, fmt.Errorf("artifact %s: %w", item.Name, err)
		}
		frames[item.Name] = rows
	}
	if source.Product == eventsProduct {
		rows := frames["solar_wind_events"]
		if len(rows) != 1 {
			return result, errors.New("expected one solar-wind event report")
		}
		raw := []byte(rows[0]["report_json"])
		var report publicforecast.SolarWindEventForecast
		if json.Unmarshal(raw, &report) != nil || report.SchemaVersion != 1 || report.Product != eventsProduct ||
			!report.IssueTime.Equal(result.issueTime) || report.SourceStatus == nil || report.Events == nil ||
			report.L1Detections == nil || !slices.Contains([]string{
			"solar-wind-events-v1", "solar-wind-events-v2", "solar-wind-events-v3", "solar-wind-events-v4",
		}, report.ModelVersion) {
			return result, errors.New("invalid solar-wind event report")
		}
		result.compact, result.withMeta = raw, raw
		return result, nil
	}
	var err error
	if source.Product == densityProduct {
		data, err := publicforecast.DensityData(frames["atmospheric_density"], result.issueTime)
		if err != nil {
			return result, err
		}
		compact := *data
		compact.Meta = nil
		result.compact, result.withMeta, err = encodeForecast(compact, data)
		return result, err
	}
	product := publicforecast.Products[source.Product]
	data, err := publicforecast.ForecastData(source.Product, product, frames, result.issueTime)
	if err != nil {
		return result, err
	}
	compact := *data
	compact.Meta = nil
	result.compact, result.withMeta, err = encodeForecast(compact, data)
	return result, err
}

func encodeForecast(compact, withMeta any) ([]byte, []byte, error) {
	left, err := json.Marshal(compact)
	if err != nil {
		return nil, nil, err
	}
	right, err := json.Marshal(withMeta)
	if err != nil {
		return nil, nil, err
	}
	return left, right, nil
}

func forecastRows(item artifact, issue time.Time) ([]publicforecast.Row, error) {
	reader := csv.NewReader(strings.NewReader(item.CSVText))
	records, err := reader.ReadAll()
	if err != nil || len(records) < 2 || len(records)-1 != item.RowCount || !slices.Equal(records[0], item.Columns) {
		return nil, errors.New("invalid forecast CSV")
	}
	fields := make(map[string]bool, len(item.Columns))
	for _, name := range item.Columns {
		if fields[name] {
			return nil, errors.New("duplicate forecast column")
		}
		fields[name] = true
	}
	rows := make([]publicforecast.Row, 0, item.RowCount)
	for _, record := range records[1:] {
		row := make(publicforecast.Row, len(item.Columns))
		for i, name := range item.Columns {
			row[name] = record[i]
		}
		at, err := publicforecast.ParseTimestamp(row["issue_time"])
		if err != nil || !at.Equal(issue) {
			return nil, errors.New("inconsistent forecast issue time")
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func (p *preparedForecasts) serve(densityMaxAge float64) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		target := r.PathValue("product")
		if !hasPreparedForecast(target) {
			detail(w, http.StatusNotFound, "Unknown forecast product")
			return
		}
		meta, ok := includeMetadata(r)
		if !ok {
			detail(w, http.StatusUnprocessableEntity, "meta must be a boolean")
			return
		}
		result, synced := p.get(target)
		body := result.compact
		if meta {
			body = result.withMeta
		}
		age := time.Since(result.issueTime).Hours()
		if !synced || body == nil || (target == densityProduct && (age < 0 || age > densityMaxAge)) {
			detail(w, http.StatusServiceUnavailable, "Forecast is not ready")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(body); err != nil {
			log.Printf("write prepared forecast: %v", err)
		}
	}
}

func includeMetadata(r *http.Request) (bool, bool) {
	switch r.URL.Query().Get("meta") {
	case "true":
		return true, true
	case "", "false":
		return false, true
	default:
		return false, false
	}
}
