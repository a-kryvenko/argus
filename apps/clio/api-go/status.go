package main

import (
	"context"
	"net/http"
	"strings"
	"time"
)

type observationStatus struct {
	ObservedAt    *time.Time `json:"observed_at"`
	ReceivedAt    *time.Time `json:"received_at"`
	SourceProduct string     `json:"source_product,omitempty"`
	Usable        bool       `json:"usable"`
}

func (api api) observationStatus(w http.ResponseWriter, r *http.Request) {
	now := time.Now().UTC()
	status, err := api.db.status(r.Context(), now)
	if err != nil {
		writeStorageError(w, "Observation status is unavailable", err, nil)
		return
	}
	writeSuccess(w, status)
}

func (db postgres) status(ctx context.Context, now time.Time) (map[string]observationStatus, error) {
	numeric, err := db.series(ctx, seriesQuery{metrics: metrics, to: now, windBasis: l1TimeBasis, windStep: time.Minute})
	if err != nil {
		return nil, err
	}
	status := make(map[string]observationStatus)
	for id, series := range numeric {
		if len(series.Points) == 0 {
			continue
		}
		point := series.Points[0]
		status[id] = observationStatus{&point.ObservedAt, point.ReceivedAt, point.SourceProduct, point.Value != nil}
	}
	rows, err := db.pool.Query(ctx, `
		SELECT DISTINCT ON (o.observation) o.observation, o.observed_at, f.timestamp, o.source_product,
		       o.details->>'source_updated_at'
		FROM clio.file_observation o JOIN clio.files f ON f.id = o.file_id
		WHERE o.source_id IS NULL AND o.observation NOT LIKE 'donki.%'
		  AND COALESCE(o.details->>'file_state', '') = ''
		ORDER BY o.observation, o.observed_at DESC, f.timestamp DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var id, product string
		var at, received time.Time
		var updated *string
		if err := rows.Scan(&id, &at, &received, &product, &updated); err != nil {
			return nil, err
		}
		at, received = at.UTC(), received.UTC()
		entry := observationStatus{&at, &received, product, true}
		if strings.HasSuffix(id, "_browse") {
			entry.ObservedAt, entry.Usable = nil, false
			if updated != nil {
				if stamp, err := time.Parse(time.RFC3339Nano, *updated); err == nil {
					stamp = stamp.UTC()
					entry.ObservedAt, entry.Usable = &stamp, true
				}
			}
		}
		status[id] = entry
	}
	return status, rows.Err()
}
