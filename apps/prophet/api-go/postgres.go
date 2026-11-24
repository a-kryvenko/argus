package main

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type postgres struct {
	pool *pgxpool.Pool
}

func (db postgres) ready(ctx context.Context) error {
	_, err := db.pool.Exec(ctx, `SELECT
		(SELECT id FROM prophet.forecast_release LIMIT 1),
		(SELECT product FROM prophet.forecast_verification_summary LIMIT 1)`)
	return err
}

func (db postgres) verification(ctx context.Context, product string) (json.RawMessage, error) {
	var report json.RawMessage
	err := db.pool.QueryRow(ctx, `SELECT report FROM prophet.forecast_verification_summary
		WHERE product=$1`, product).Scan(&report)
	if err != nil {
		return nil, err
	}
	var document struct {
		Product  string
		Protocol string
		Groups   []json.RawMessage
	}
	if err := json.Unmarshal(report, &document); err != nil || document.Product != product ||
		document.Protocol != "published-hourly-v1" || document.Groups == nil {
		return nil, fmt.Errorf("invalid prepared verification report for %s", product)
	}
	return report, nil
}

type solarIndices struct {
	ContractVersion int                `json:"contract_version"`
	AsOf            time.Time          `json:"as_of"`
	Items           []solarMeasurement `json:"items"`
}

type solarMeasurement struct {
	Metric     string    `json:"metric"`
	ObservedAt time.Time `json:"observed_at"`
	Value      *float64  `json:"value"`
}

func (db postgres) solarIndices(ctx context.Context) (solarIndices, error) {
	result := solarIndices{ContractVersion: 1, Items: []solarMeasurement{}}
	var raw []byte
	err := db.pool.QueryRow(ctx, `SELECT (input_snapshot->>'as_of')::timestamptz,input_snapshot->'measurements'
		FROM prophet.forecast_run WHERE input_snapshot IS NOT NULL
		AND (product IN ('solar-radiation','atmospheric-density')
		     OR input_snapshot->'files' @> '[{"observation":"goes"}]'::jsonb)
		ORDER BY started_at DESC,id DESC LIMIT 1`).Scan(&result.AsOf, &raw)
	if err != nil {
		return result, err
	}
	var items []struct {
		Metric     string   `json:"metric"`
		ObservedAt string   `json:"observed_at"`
		Value      *float64 `json:"value"`
	}
	if err := json.Unmarshal(raw, &items); err != nil || items == nil {
		return result, fmt.Errorf("invalid solar measurements")
	}
	cutoff := result.AsOf.Add(-14 * 24 * time.Hour)
	for _, item := range items {
		switch item.Metric {
		case "f10_7", "s10", "m10", "y10":
			observed, err := time.Parse(time.RFC3339Nano, item.ObservedAt)
			if err != nil {
				return result, err
			}
			if observed.Before(cutoff) || observed.After(result.AsOf) {
				continue
			}
			if item.Value == nil {
				return result, fmt.Errorf("missing solar measurement value")
			}
			result.Items = append(result.Items, solarMeasurement{item.Metric, observed, item.Value})
		}
	}
	return result, nil
}
