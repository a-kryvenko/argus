package main

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type measurement struct {
	metric     string
	observedAt time.Time
	value      *float64
}

type postgres struct {
	pool    *pgxpool.Pool
	sources map[string]sourcePolicy
}

func (db postgres) ready(ctx context.Context) error {
	_, err := db.pool.Exec(ctx, `SELECT m.spacecraft, m.quality, n.wind_time_basis, o.id, f.id
		FROM clio.measurement m, clio.normalized_observation n, clio.file_observation o, clio.files f LIMIT 0`)
	return err
}

func (db postgres) latest(ctx context.Context, metrics []string) ([]measurement, error) {
	series, err := db.series(ctx, seriesQuery{metrics: metrics, to: time.Now().UTC(), windBasis: l1TimeBasis, windStep: time.Minute})
	if err != nil {
		return nil, err
	}
	result := make([]measurement, 0, len(series))
	for metric, readings := range series {
		for _, point := range readings.Points {
			result = append(result, measurement{metric: metric, observedAt: point.ObservedAt, value: point.Value})
		}
	}
	return result, nil
}
