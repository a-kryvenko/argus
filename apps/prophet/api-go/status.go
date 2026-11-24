package main

import (
	"context"
	"encoding/json"
	"time"

	"github.com/jackc/pgx/v5"
)

type forecastStatus struct {
	ContractVersion int               `json:"contract_version"`
	Product         string            `json:"product"`
	AssessedAt      time.Time         `json:"assessed_at"`
	Mode            string            `json:"mode"`
	CurrentRelease  *currentRelease   `json:"current_release"`
	ReleaseAgeHours *float64          `json:"release_age_hours"`
	MaxAgeHours     *float64          `json:"existing_public_max_age_hours"`
	Freshness       string            `json:"freshness"`
	LatestAttempt   *forecastAttempt  `json:"latest_attempt"`
	Artifacts       []attemptArtifact `json:"latest_attempt_artifacts"`
	Note            string            `json:"note"`
}

type currentRelease struct {
	ReleaseID        string          `json:"release_id"`
	RunID            string          `json:"run_id"`
	IssueTime        time.Time       `json:"issue_time"`
	PublishedAt      time.Time       `json:"published_at"`
	StartedAt        time.Time       `json:"started_at"`
	InputDiagnostics json.RawMessage `json:"input_diagnostics"`
}

type forecastAttempt struct {
	RunID            string          `json:"run_id"`
	Status           string          `json:"status"`
	StartedAt        time.Time       `json:"started_at"`
	FinishedAt       *time.Time      `json:"finished_at"`
	Error            *string         `json:"error"`
	InputDiagnostics json.RawMessage `json:"input_diagnostics"`
	Publication      json.RawMessage `json:"publication"`
}

type attemptArtifact struct {
	Name   string  `json:"name"`
	Status string  `json:"status"`
	Error  *string `json:"error"`
}

const statusQuery = `
WITH catalog AS (
	SELECT entry->>'name' AS name, entry->>'historical_name' AS historical_name,
	       entry->'artifacts' AS artifacts, ordinal
	FROM jsonb_array_elements($1::jsonb) WITH ORDINALITY AS products(entry, ordinal)
), current_releases AS (
	SELECT r.product, r.id AS release_id, r.run_id, r.issue_time, r.published_at,
	       f.started_at, f.provenance->'input_diagnostics' AS input_diagnostics
	FROM (` + latestReleases + `) r
	JOIN prophet.forecast_run f ON f.id = r.run_id
)
SELECT p.name, to_jsonb(current_release) - 'product', to_jsonb(attempt),
       COALESCE(artifacts.data, '[]'::jsonb)
FROM catalog p
LEFT JOIN current_releases current_release ON current_release.product = p.name
LEFT JOIN LATERAL (
	SELECT f.id AS run_id, f.status, f.started_at, f.finished_at, f.error,
	       f.provenance->'input_diagnostics' AS input_diagnostics,
	       f.provenance->'publication' AS publication
	FROM prophet.forecast_run f
	WHERE f.product IN ('all', p.name, p.historical_name)
	ORDER BY f.started_at DESC, f.id DESC
	LIMIT 1
) attempt ON true
LEFT JOIN LATERAL (
	SELECT jsonb_agg(jsonb_build_object('name', a.name, 'status', a.status, 'error', a.error)
	                 ORDER BY a.name) AS data
	FROM prophet.forecast_artifact a
	WHERE a.run_id = attempt.run_id AND p.artifacts ? a.name
) artifacts ON true
ORDER BY p.ordinal`

func (db postgres) status(ctx context.Context, now time.Time, densityMaxAge float64) ([]forecastStatus, error) {
	rows, err := db.pool.Query(ctx, statusQuery, products)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(row pgx.CollectableRow) (forecastStatus, error) {
		item := forecastStatus{
			ContractVersion: 1, AssessedAt: now, Mode: "observe", Freshness: "unavailable",
			Note: "Diagnostic status does not establish model readiness or individual sensor freshness.",
		}
		if err := row.Scan(&item.Product, &item.CurrentRelease, &item.LatestAttempt, &item.Artifacts); err != nil {
			return forecastStatus{}, err
		}
		if item.Product == "atmospheric-density" {
			item.MaxAgeHours = &densityMaxAge
		}
		if item.CurrentRelease != nil {
			age := now.Sub(item.CurrentRelease.IssueTime).Hours()
			item.ReleaseAgeHours = &age
			item.Freshness = releaseFreshness(age, item.MaxAgeHours)
		}
		return item, nil
	})
}

func releaseFreshness(age float64, maxAge *float64) string {
	switch {
	case age < 0:
		return "future_issue_time"
	case maxAge == nil:
		return "unconfigured"
	case age > *maxAge:
		return "stale"
	default:
		return "within_age_limit"
	}
}
