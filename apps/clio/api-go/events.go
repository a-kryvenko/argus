package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
)

var corruptEvents = errors.New("Normalized DONKI observations are unavailable or corrupt")
var donkiEndpoints = []string{"CME", "WSAEnlilSimulations", "HSS", "IPS"}
var donkiObservations = []string{"donki.CME", "donki.WSAEnlilSimulations", "donki.HSS", "donki.IPS"}

type eventInputs struct {
	SchemaVersion  int                    `json:"schema_version"`
	ReadAt         apiTime                `json:"read_at"`
	Polls          []sourcePoll           `json:"polls"`
	Events         []solarWindEvent       `json:"events"`
	EnlilArrivals  []enlilArrival         `json:"enlil_arrivals"`
	LatestAttempts map[string]pollAttempt `json:"latest_attempts"`
}

type sourcePoll struct {
	Endpoint   string  `json:"endpoint"`
	ReceivedAt apiTime `json:"received_at"`
	SHA256     string  `json:"sha256"`
	pollDetails
}

type pollDetails struct {
	PollID       pgtype.UUID `json:"poll_id"`
	RequestStart *apiTime    `json:"request_start"`
	RequestEnd   *apiTime    `json:"request_end"`
	RecordCount  *int        `json:"record_count"`
}

type pollAttempt struct {
	ReceivedAt    string  `json:"received_at"`
	Error         *string `json:"error"`
	LastSuccessAt string  `json:"last_success_at"`
}

type arrivalReceipt struct {
	PollID     pgtype.UUID `json:"poll_id"`
	SHA256     string      `json:"sha256"`
	ReceivedAt *apiTime    `json:"received_at"`
}

type solarWindEvent struct {
	EventID            *string          `json:"event_id"`
	Kind               string           `json:"kind"`
	Status             string           `json:"status"`
	Source             string           `json:"source"`
	SourceIDs          []string         `json:"source_ids"`
	SourceURL          *string          `json:"source_url"`
	AvailableAt        *apiTime         `json:"available_at"`
	ObservedAt         *apiTime         `json:"observed_at"`
	ModelCompletedAt   *apiTime         `json:"model_completed_at"`
	ArrivalAt          *apiTime         `json:"arrival_at"`
	ArrivalTarget      *string          `json:"arrival_target"`
	DurationHours      *float64         `json:"duration_hours"`
	GlancingBlow       *bool            `json:"glancing_blow"`
	MinorImpact        *bool            `json:"minor_impact"`
	SimulationID       *string          `json:"simulation_id"`
	InitialSpeedKmS    *float64         `json:"initial_speed_km_s"`
	ArrivalModel       *string          `json:"arrival_model"`
	ArrivalMethod      *string          `json:"arrival_method"`
	ArrivalReceipts    []arrivalReceipt `json:"arrival_receipts"`
	ArrivalWindowStart *apiTime         `json:"arrival_window_start"`
	ArrivalWindowEnd   *apiTime         `json:"arrival_window_end"`
}

type enlilArrival struct {
	SimulationID     *string          `json:"simulation_id"`
	CMEIDs           []string         `json:"cme_ids"`
	AvailableAt      *apiTime         `json:"available_at"`
	ModelCompletedAt *apiTime         `json:"model_completed_at"`
	EarthArrivalAt   *apiTime         `json:"earth_arrival_at"`
	SourceURL        *string          `json:"source_url"`
	DurationHours    *float64         `json:"duration_hours"`
	GlancingBlow     *bool            `json:"glancing_blow"`
	MinorImpact      *bool            `json:"minor_impact"`
	Receipts         []arrivalReceipt `json:"receipts"`
	InvalidReason    *string          `json:"invalid_reason"`
}

func (api api) events(w http.ResponseWriter, r *http.Request) {
	inputs, err := api.db.eventInputs(r.Context(), time.Now().UTC())
	if err != nil {
		writeStorageError(w, "Event observation storage is unavailable", err, corruptEvents)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, inputs)
}

func (db postgres) eventInputs(ctx context.Context, now time.Time) (eventInputs, error) {
	tx, err := db.pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return eventInputs{}, err
	}
	defer func() {
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_ = tx.Rollback(cleanup)
	}()
	if _, err := tx.Exec(ctx, "SET LOCAL statement_timeout = '30s'"); err != nil {
		return eventInputs{}, err
	}

	inputs := eventInputs{SchemaVersion: 2, ReadAt: apiTime{now}, LatestAttempts: make(map[string]pollAttempt)}
	inputs.Polls, err = readPolls(ctx, tx)
	if err != nil {
		return eventInputs{}, err
	}
	for _, poll := range inputs.Polls {
		stamp := formatTime(poll.ReceivedAt.Time, false)
		inputs.LatestAttempts[poll.Endpoint] = pollAttempt{ReceivedAt: stamp, LastSuccessAt: stamp}
	}
	const eventsQuery = `SELECT details -> 'data' FROM clio.file_observation
		WHERE observation = ANY($1::varchar[]) AND details ->> 'kind' = 'event'
		ORDER BY observed_at, source_id`
	inputs.Events, err = readEventRows(ctx, tx, eventsQuery, donkiObservations, parseSolarWindEvent)
	if err != nil {
		return eventInputs{}, err
	}
	const arrivalsQuery = `SELECT details -> 'data' FROM clio.file_observation
		WHERE observation = 'donki.WSAEnlilSimulations' AND details ->> 'kind' = 'arrival'
		ORDER BY source_id`
	inputs.EnlilArrivals, err = readEventRows(ctx, tx, arrivalsQuery, nil, parseEnlilArrival)
	return inputs, err
}

func readPolls(ctx context.Context, tx pgx.Tx) ([]sourcePoll, error) {
	rows, err := tx.Query(ctx, `
		SELECT source_product, observed_at, sha256, details
		FROM clio.file_observation
		WHERE observation = ANY($1::varchar[]) AND source_id = ''
		ORDER BY source_product
	`, donkiObservations)
	if err != nil {
		return nil, err
	}
	polls, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (sourcePoll, error) {
		var poll sourcePoll
		var details []byte
		if err := row.Scan(&poll.Endpoint, &poll.ReceivedAt.Time, &poll.SHA256, &details); err != nil {
			return poll, err
		}
		poll.ReceivedAt.Time = poll.ReceivedAt.UTC()
		var metadata struct {
			pollDetails
			Count json.RawMessage `json:"record_count"`
		}
		if err := json.Unmarshal(details, &metadata); err != nil {
			return poll, fmt.Errorf("%w: %v", corruptEvents, err)
		}
		poll.pollDetails = metadata.pollDetails
		if !slices.Contains(donkiEndpoints, poll.Endpoint) || !sha256Pattern.MatchString(poll.SHA256) ||
			!poll.PollID.Valid || poll.RequestStart == nil || poll.RequestEnd == nil || len(metadata.Count) == 0 {
			return poll, corruptEvents
		}
		if err := json.Unmarshal(metadata.Count, &poll.RecordCount); err != nil ||
			(poll.RecordCount != nil && *poll.RecordCount < 0) {
			return poll, corruptEvents
		}
		return poll, nil
	})
	return polls, err
}

func readEventRows[T any](ctx context.Context, tx pgx.Tx, query string, observations []string, parse func([]byte) (T, error)) ([]T, error) {
	var args []any
	if observations != nil {
		args = append(args, observations)
	}
	rows, err := tx.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(row pgx.CollectableRow) (T, error) {
		var data []byte
		if err := row.Scan(&data); err != nil {
			var zero T
			return zero, err
		}
		return parse(data)
	})
}

func parseSolarWindEvent(data []byte) (solarWindEvent, error) {
	event := solarWindEvent{SourceIDs: []string{}, ArrivalReceipts: []arrivalReceipt{}}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&event); err != nil {
		return event, fmt.Errorf("%w: %v", corruptEvents, err)
	}
	if event.EventID == nil || event.AvailableAt == nil || event.SourceIDs == nil || !validReceipts(event.ArrivalReceipts) {
		return event, corruptEvents
	}
	switch event.Kind {
	case "cme", "hss", "interplanetary_shock", "solar_wind_disturbance":
	default:
		return event, corruptEvents
	}
	switch event.Status {
	case "detected", "predicted_hit", "no_predicted_hit", "prediction_elapsed", "observed":
	default:
		return event, corruptEvents
	}
	switch event.Source {
	case "DONKI", "DONKI/WSA-ENLIL", "ARGUS":
	default:
		return event, corruptEvents
	}
	if (event.ArrivalTarget != nil && *event.ArrivalTarget != "Earth") ||
		(event.DurationHours != nil && *event.DurationHours <= 0) ||
		(event.InitialSpeedKmS != nil && *event.InitialSpeedKmS <= 0) {
		return event, corruptEvents
	}
	start, end := event.ArrivalWindowStart, event.ArrivalWindowEnd
	if (start != nil || end != nil) &&
		(start == nil || end == nil || event.ArrivalAt == nil || event.ArrivalAt.Before(start.Time) || event.ArrivalAt.After(end.Time)) {
		return event, corruptEvents
	}
	return event, nil
}

func parseEnlilArrival(data []byte) (enlilArrival, error) {
	arrival := enlilArrival{CMEIDs: []string{}, Receipts: []arrivalReceipt{}}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&arrival); err != nil {
		return arrival, fmt.Errorf("%w: %v", corruptEvents, err)
	}
	if arrival.SimulationID == nil || arrival.AvailableAt == nil || arrival.ModelCompletedAt == nil ||
		arrival.CMEIDs == nil || !validReceipts(arrival.Receipts) ||
		(arrival.DurationHours != nil && *arrival.DurationHours <= 0) {
		return arrival, corruptEvents
	}
	return arrival, nil
}

func validReceipts(receipts []arrivalReceipt) bool {
	if receipts == nil {
		return false
	}
	for _, receipt := range receipts {
		if !receipt.PollID.Valid || !sha256Pattern.MatchString(receipt.SHA256) || receipt.ReceivedAt == nil {
			return false
		}
	}
	return true
}
