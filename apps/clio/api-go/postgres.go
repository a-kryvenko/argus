package main

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type measurement struct {
	metric     string
	observedAt time.Time
	value      *float64
}

type postgres struct {
	pool *pgxpool.Pool
}

func (db postgres) ready(ctx context.Context) error {
	_, err := db.pool.Exec(ctx, "SELECT 1 FROM clio.measurement LIMIT 1")
	return err
}

func (db postgres) latest(ctx context.Context, metrics []string) ([]measurement, error) {
	rows, err := db.pool.Query(ctx, latestMeasurements, metrics, windMetrics, nativeWindProducts)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, scanMeasurement)
}

const latestMeasurements = `
		SELECT DISTINCT ON (metric) metric, observed_at, value
		FROM clio.measurement
		WHERE metric = ANY($1::varchar[])
		  AND (metric <> ALL($2::varchar[]) OR source_product = ANY($3::varchar[]))
		ORDER BY metric, observed_at DESC
	`

func scanMeasurement(row pgx.CollectableRow) (measurement, error) {
	var reading measurement
	err := row.Scan(&reading.metric, &reading.observedAt, &reading.value)
	return reading, err
}
