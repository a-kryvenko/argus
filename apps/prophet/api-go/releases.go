package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"time"
	"unicode/utf8"

	"github.com/jackc/pgx/v5"
)

var products = []struct {
	Name           string   `json:"name"`
	Artifacts      []string `json:"artifacts"`
	HistoricalName string   `json:"historical_name,omitempty"`
}{
	{"solar-wind-events", []string{"solar_wind_events"}, ""},
	{"solar-wind-speed", []string{"plasma_speed_quantile", "plasma_speed_threshold"}, "wind"},
	{"solar-wind-density", []string{"plasma_density_quantile"}, ""},
	{"solar-wind-temperature", []string{"plasma_temperature_quantile"}, ""},
	{hmfProduct, []string{"hmf_total_threshold", "hmf_southward_threshold"}, ""},
	{"solar-radiation", []string{"f10_7_quantile", "s10_quantile", "m10_quantile", "y10_quantile"}, ""},
	{geomagneticProduct, []string{"kp_threshold", "ap_quantile"}, "kp"},
	{"dst", []string{"dst_quantile"}, ""},
	{"atmospheric-density", []string{"atmospheric_density"}, "density"},
}

func productArtifacts(product string) []string {
	if spec := fastForecastSpecFor(product); spec != nil {
		return []string{spec.artifact}
	}
	for _, candidate := range products {
		if candidate.Name == product {
			return candidate.Artifacts
		}
	}
	return nil
}

const latestReleases = `SELECT DISTINCT ON (r.product) r.*
	FROM prophet.forecast_release r JOIN prophet.forecast_run f ON f.id=r.run_id
	ORDER BY r.product,r.issue_time DESC,f.started_at DESC,r.published_at DESC,r.id DESC`

const maxArtifactBytes = 32 * 1024 * 1024

type release struct {
	ContractVersion int        `json:"contract_version"`
	ID              string     `json:"release_id"`
	RunID           string     `json:"run_id"`
	Product         string     `json:"product"`
	PublishedAt     time.Time  `json:"published_at"`
	IssueTime       time.Time  `json:"issue_time"`
	Artifacts       []artifact `json:"artifacts"`
}

type artifact struct {
	Name      string          `json:"name"`
	SHA256    string          `json:"sha256"`
	RowCount  int             `json:"row_count"`
	Columns   []string        `json:"columns"`
	CSVText   string          `json:"csv_text"`
	ModelInfo json.RawMessage `json:"model_info"`
}

func (db postgres) readRelease(ctx context.Context, product, id string) (release, error) {
	result := release{ContractVersion: 1, Product: product}
	var names []string
	tx, err := db.pool.BeginTx(ctx, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly})
	if err != nil {
		return result, err
	}
	defer tx.Rollback(ctx)
	var row pgx.Row
	const columns = "r.id::text,r.run_id::text,r.published_at,r.issue_time,r.artifact_names"
	if id == "" {
		row = tx.QueryRow(ctx, "SELECT "+columns+" FROM ("+latestReleases+") r WHERE r.product=$1", product)
	} else {
		row = tx.QueryRow(ctx, "SELECT "+columns+" FROM prophet.forecast_release r WHERE r.product=$1 AND r.id=$2::text::uuid", product, id)
	}
	if err := row.Scan(&result.ID, &result.RunID, &result.PublishedAt, &result.IssueTime, &names); err != nil {
		return result, err
	}
	expected := productArtifacts(product)
	slices.Sort(names)
	ordered := slices.Clone(expected)
	slices.Sort(ordered)
	if !slices.Equal(names, ordered) {
		return result, fmt.Errorf("incomplete release for %s", product)
	}
	rows, err := tx.Query(ctx, `SELECT name,sha256,row_count,columns,csv_gzip,model_info
		FROM prophet.forecast_artifact
		WHERE run_id=$1::text::uuid AND name=ANY($2::varchar[]) AND status='stored' ORDER BY name`, result.RunID, names)
	if err != nil {
		return result, err
	}
	type storedArtifact struct {
		artifact
		compressed []byte
	}
	stored, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (storedArtifact, error) {
		var item storedArtifact
		err := row.Scan(&item.Name, &item.SHA256, &item.RowCount, &item.Columns, &item.compressed, &item.ModelInfo)
		return item, err
	})
	if err != nil {
		return result, err
	}
	if len(stored) != len(names) {
		return result, fmt.Errorf("incomplete artifacts for %s", product)
	}
	if err := tx.Commit(ctx); err != nil {
		return result, err
	}
	for _, item := range stored {
		if err := item.decode(item.compressed); err != nil {
			return result, fmt.Errorf("artifact %s: %w", item.Name, err)
		}
		result.Artifacts = append(result.Artifacts, item.artifact)
	}
	return result, nil
}

func (item *artifact) decode(compressed []byte) error {
	reader, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		return err
	}
	defer reader.Close()
	content, err := io.ReadAll(io.LimitReader(reader, maxArtifactBytes+1))
	if err != nil {
		return err
	}
	if len(content) > maxArtifactBytes || !utf8.Valid(content) || fmt.Sprintf("%x", sha256.Sum256(content)) != item.SHA256 {
		return fmt.Errorf("invalid artifact size, encoding or checksum")
	}
	var metadata map[string]json.RawMessage
	if err := json.Unmarshal(item.ModelInfo, &metadata); err != nil || metadata == nil {
		return fmt.Errorf("invalid artifact metadata")
	}
	header, err := csv.NewReader(bytes.NewReader(content)).Read()
	if err != nil || !slices.Equal(header, item.Columns) || item.RowCount < 1 {
		return fmt.Errorf("invalid artifact columns or row count")
	}
	item.CSVText = string(content)
	return nil
}
