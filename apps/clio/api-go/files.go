package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strconv"
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
var sdoSizes = []int{128, 256, 512, 1024, 2048, 4096}
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
	observations := make([]string, 0, len(sdoChannels))
	for _, channel := range sdoChannels {
		observations = append(observations, "aia_"+channel+"_browse")
	}
	rows, err := db.pool.Query(ctx, `
		SELECT DISTINCT ON (o.observation) o.observation, o.details, f.key
		FROM clio.file_observation AS o
		JOIN clio.files AS f ON f.id = o.file_id
		WHERE o.observation = ANY($1::varchar[])
		ORDER BY o.observation, o.observed_at DESC, o.id DESC
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
		if details == nil {
			continue
		}
		var metadata map[string]json.RawMessage
		if err := json.Unmarshal(details, &metadata); err != nil {
			return nil, err
		}
		var channel string
		if err := json.Unmarshal(metadata["channel"], &channel); err != nil || !slices.Contains(sdoChannels, channel) {
			continue
		}
		if _, seen := images[channel]; seen {
			continue
		}
		if !strings.HasPrefix(base, "https://") && !strings.HasPrefix(base, "http://") {
			return nil, errors.New("S3_PUBLIC_BASE_URL must be the public bucket origin")
		}
		if observation != "aia_"+channel+"_browse" {
			return nil, errors.New("SDO image channel does not match its observation")
		}
		var image sdoImage
		if err := json.Unmarshal(details, &image.sdoMetadata); err != nil {
			return nil, err
		}
		if image.CapturedAt == nil || !assetPattern.MatchString(image.AssetID) {
			return nil, fmt.Errorf("invalid SDO image metadata for channel %s", channel)
		}
		image.URLs = sdoImageURLs(base, key)
		images[channel] = image
	}
	ordered := make([]sdoImage, 0, len(images))
	for _, channel := range sdoChannels {
		if image, exists := images[channel]; exists {
			ordered = append(ordered, image)
		}
	}
	return ordered, rows.Err()
}

func sdoImageURLs(base, key string) map[string]string {
	prefix := key
	if lastSlash := strings.LastIndexByte(key, '/'); lastSlash >= 0 {
		prefix = key[:lastSlash]
	}
	escaped := strings.NewReplacer("+", "%20", "%2F", "/").Replace(url.QueryEscape(prefix))
	root := base + "/" + escaped + "/"
	urls := map[string]string{"jpg.original": root + "original.jpg"}
	for _, size := range sdoSizes {
		width := strconv.Itoa(size)
		if size != 4096 {
			urls["jpg."+width] = root + width + ".jpg"
		}
		urls["webp."+width] = root + width + ".webp"
	}
	return urls
}
