package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strings"
	"time"
)

const maxFiles = 10_000

var fileOverflow = errors.New("Observation file range exceeds the supported batch size")
var sha256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

type fileReference struct {
	ID        string  `json:"id"`
	Timestamp apiTime `json:"timestamp"`
	Bucket    string  `json:"bucket"`
	Key       string  `json:"key"`
}

type observationFile struct {
	Observation   string        `json:"observation"`
	ObservedAt    apiTime       `json:"observed_at"`
	SHA256        string        `json:"sha256"`
	SourceProduct string        `json:"source_product"`
	File          fileReference `json:"file"`
}

func (api api) files(w http.ResponseWriter, r *http.Request) {
	validation := queryValidation{values: requestQuery(r)}
	observation, _ := validation.required("observation")
	start := validation.dateTime("start")
	end := validation.dateTime("end")
	if !validation.valid(w) {
		return
	}
	if !validRange(start, end) {
		writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Invalid observation file interval")
		return
	}
	files, err := api.db.files(r.Context(), observation, start, end)
	if err != nil {
		writeStorageError(w, "Observation metadata is unavailable", err, fileOverflow)
		return
	}
	writeSuccess(w, files)
}

func (db postgres) files(ctx context.Context, observation string, start, end time.Time) ([]observationFile, error) {
	rows, err := db.pool.Query(ctx, `
		SELECT o.observation, o.observed_at, o.sha256, o.source_product,
		       f.id::text, f.timestamp, f.bucket, f.key
		FROM clio.file_observation AS o
		JOIN clio.files AS f ON f.id = o.file_id
		WHERE o.observation = $1 AND (o.source_id IS NULL OR o.source_id = '')
		  AND COALESCE(o.details ->> 'file_state', '') = ''
		  AND o.observed_at >= $2 AND o.observed_at < $3
		ORDER BY o.observed_at, o.source_product, f.key
		LIMIT $4
	`, observation, start, end, maxFiles+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	files := make([]observationFile, 0)
	for rows.Next() {
		if len(files) == maxFiles {
			return nil, fileOverflow
		}
		var item observationFile
		err := rows.Scan(&item.Observation, &item.ObservedAt.Time, &item.SHA256, &item.SourceProduct,
			&item.File.ID, &item.File.Timestamp.Time, &item.File.Bucket, &item.File.Key)
		if err != nil {
			return nil, err
		}
		item.ObservedAt.Time = item.ObservedAt.UTC()
		item.File.Timestamp.Time = item.File.Timestamp.UTC()
		files = append(files, item)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for _, item := range files {
		if item.Observation == "" || !sha256Pattern.MatchString(item.SHA256) {
			return nil, errors.New("invalid observation or SHA-256 in file metadata")
		}
	}
	return files, nil
}

var sdoChannels = []string{"094", "131", "171", "193", "211", "304", "335", "1600", "1700"}
var sdoVariantPattern = regexp.MustCompile(`^([1-9][0-9]{1,3})\.(jpg|webp)$`)
var assetPattern = regexp.MustCompile(`^[0-9]{10}_(` + strings.Join(sdoChannels, "|") + `)_[0-9a-f]{64}$`)

type sdoMetadata struct {
	Channel         string   `json:"channel"`
	AssetID         string   `json:"asset_id"`
	CapturedAt      *apiTime `json:"captured_at"`
	SourceUpdatedAt *apiTime `json:"source_updated_at"`
}

type sdoImage struct {
	sdoMetadata
	URLs map[string]string `json:"urls"`
}

func (api api) sdo(w http.ResponseWriter, r *http.Request) {
	images, err := api.db.sdoImages(r.Context(), api.publicBase)
	if err != nil {
		writeStorageError(w, "SDO image storage is unavailable", err, nil)
		return
	}
	writeSuccess(w, map[string]any{"images": images})
}

func (db postgres) sdoImages(ctx context.Context, base string) ([]sdoImage, error) {
	origin, err := url.Parse(base)
	if err != nil || (origin.Scheme != "https" && origin.Scheme != "http") || origin.Host == "" ||
		origin.RawQuery != "" || origin.Fragment != "" || origin.User != nil {
		return nil, errors.New("S3_PUBLIC_BASE_URL must be the public bucket origin")
	}
	observations := make([]string, 0, len(sdoChannels))
	for _, channel := range sdoChannels {
		observations = append(observations, "aia_"+channel+"_browse")
	}
	rows, err := db.pool.Query(ctx, `
		SELECT wanted.observation, latest.details, latest.key
		FROM unnest($1::varchar[]) AS wanted(observation)
		CROSS JOIN LATERAL (
			SELECT o.details, f.key, o.observed_at, o.id
			FROM clio.file_observation o
			JOIN clio.files f ON f.id = o.file_id
			WHERE o.observation = wanted.observation AND o.source_id IS NULL
			  AND f.bucket = 'argus-public' AND COALESCE(o.details ->> 'file_state', '') = ''
			ORDER BY o.observed_at DESC, o.id DESC
			LIMIT 8
		) latest
		ORDER BY wanted.observation, latest.observed_at DESC, latest.id DESC
	`, observations)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	images := make(map[string]sdoImage)
	for rows.Next() {
		var observation, key string
		var details []byte
		if err := rows.Scan(&observation, &details, &key); err != nil {
			return nil, err
		}
		channel := strings.TrimSuffix(strings.TrimPrefix(observation, "aia_"), "_browse")
		if _, seen := images[channel]; seen {
			continue
		}
		var metadata struct {
			sdoMetadata
			ExtraFiles []string `json:"extra_files"`
		}
		matches := []string(nil)
		if err := json.Unmarshal(details, &metadata); err == nil {
			matches = assetPattern.FindStringSubmatch(metadata.AssetID)
		}
		if metadata.Channel != channel || len(matches) != 2 || matches[1] != channel || path.Base(key) != "original.jpg" {
			log.Printf("Ignoring invalid SDO browse manifest for %s", observation)
			continue
		}
		images[channel] = sdoImage{sdoMetadata: metadata.sdoMetadata, URLs: sdoImageURLs(base, key, metadata.ExtraFiles)}
	}
	ordered := make([]sdoImage, 0, len(images))
	for _, channel := range sdoChannels {
		if image, exists := images[channel]; exists {
			ordered = append(ordered, image)
		}
	}
	return ordered, rows.Err()
}

func sdoImageURLs(base, key string, variants []string) map[string]string {
	escape := strings.NewReplacer("+", "%20", "%2F", "/")
	objectURL := func(objectKey string) string {
		return base + "/" + escape.Replace(url.QueryEscape(objectKey))
	}
	urls := map[string]string{"jpg.original": objectURL(key)}
	for _, variant := range variants {
		match := sdoVariantPattern.FindStringSubmatch(path.Base(variant))
		if path.Dir(variant) == path.Dir(key) && len(match) == 3 {
			urls[match[2]+"."+match[1]] = objectURL(variant)
		}
	}
	return urls
}
