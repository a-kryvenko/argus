package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
)

const statusTimeout = 5 * time.Second

type reader interface {
	ready(context.Context) error
	readRelease(context.Context, string, string) (release, error)
	status(context.Context, time.Time, float64) ([]forecastStatus, error)
	solarIndices(context.Context) (solarIndices, error)
	verification(context.Context, string) (json.RawMessage, error)
}

func newHandler(db reader, token, summaryPath string, densityMaxAge float64) http.Handler {
	mux := http.NewServeMux()
	register := func(path string, handler http.HandlerFunc) {
		mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodGet {
				w.Header().Set("Allow", http.MethodGet)
				detail(w, http.StatusMethodNotAllowed, "Method Not Allowed")
				return
			}
			handler(w, r)
		})
	}
	protected := func(path string, handler http.HandlerFunc) {
		register(path, func(w http.ResponseWriter, r *http.Request) {
			if token == "" {
				detail(w, http.StatusServiceUnavailable, "Forecast read service is not configured")
				return
			}
			scheme, credentials, _ := strings.Cut(r.Header.Get("Authorization"), " ")
			if !strings.EqualFold(scheme, "Bearer") || subtle.ConstantTimeCompare([]byte(credentials), []byte(token)) != 1 {
				detail(w, http.StatusUnauthorized, "Invalid service credentials")
				return
			}
			handler(w, r)
		})
	}
	health := func(w http.ResponseWriter) {
		writeJSON(w, 200, map[string]string{"service": "prophet", "status": "ok"})
	}
	register("/health/live", func(w http.ResponseWriter, r *http.Request) { health(w) })
	register("/health/ready", func(w http.ResponseWriter, r *http.Request) {
		if token == "" {
			detail(w, 503, "Forecast read service is not configured")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()
		if err := db.ready(ctx); err != nil {
			unavailable(w, "Forecast storage is not ready", err)
			return
		}
		health(w)
	})
	readRelease := func(w http.ResponseWriter, r *http.Request, id string) {
		product := r.PathValue("product")
		if productArtifacts(product) == nil {
			detail(w, 404, "Unknown forecast product")
			return
		}
		data, err := db.readRelease(r.Context(), product, id)
		switch {
		case errors.Is(err, pgx.ErrNoRows):
			code := http.StatusServiceUnavailable
			if id != "" {
				code = http.StatusNotFound
			}
			detail(w, code, "Forecast release is not available")
		case err != nil:
			unavailable(w, "Forecast storage is not ready", err)
		default:
			writeJSON(w, 200, data)
		}
	}
	protected("/internal/v1/forecasts/{product}/latest", func(w http.ResponseWriter, r *http.Request) {
		readRelease(w, r, "")
	})
	protected("/internal/v1/forecasts/{product}/releases/{release_id}", func(w http.ResponseWriter, r *http.Request) {
		input := r.PathValue("release_id")
		value := input
		if len(value) >= 9 && strings.EqualFold(value[:9], "urn:uuid:") {
			value = value[9:]
		}
		if strings.HasPrefix(value, "{") && strings.HasSuffix(value, "}") {
			value = value[1 : len(value)-1]
		}
		wrongSeparators := len(value) == 36 && (value[8] != '-' || value[13] != '-' || value[18] != '-' || value[23] != '-')
		var id pgtype.UUID
		if err := id.Scan(value); err != nil || wrongSeparators {
			writeJSON(w, 422, map[string]any{"detail": []map[string]any{{
				"type": "uuid_parsing", "loc": []string{"path", "release_id"},
				"msg": "Input should be a valid UUID", "input": input,
			}}})
			return
		}
		readRelease(w, r, id.String())
	})
	protected("/internal/v1/forecasts/status", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), statusTimeout)
		defer cancel()
		data, err := db.status(ctx, time.Now().UTC(), densityMaxAge)
		if err != nil {
			unavailable(w, "Forecast storage is not ready", err)
			return
		}
		writeJSON(w, 200, struct {
			ContractVersion int              `json:"contract_version"`
			Products        []forecastStatus `json:"products"`
		}{1, data})
	})
	protected("/internal/v1/solar-indices", func(w http.ResponseWriter, r *http.Request) {
		data, err := db.solarIndices(r.Context())
		switch {
		case errors.Is(err, pgx.ErrNoRows):
			detail(w, 503, "Solar-index observations are not recorded yet")
		case err != nil:
			unavailable(w, "Solar indices are unavailable", err)
		default:
			writeJSON(w, 200, data)
		}
	})
	protected("/internal/v1/forecasts/{product}/verification", func(w http.ResponseWriter, r *http.Request) {
		product := r.PathValue("product")
		if product == "solar-wind-events" {
			detail(w, 404, "Event verification is not available yet")
			return
		}
		if productArtifacts(product) == nil {
			detail(w, 404, "Unknown forecast product")
			return
		}
		data, err := db.verification(r.Context(), product)
		if err != nil {
			unavailable(w, "Forecast verification is unavailable", err)
			return
		}
		writeJSON(w, 200, data)
	})
	protected("/internal/v1/summary", func(w http.ResponseWriter, r *http.Request) {
		content, err := os.ReadFile(summaryPath)
		if err != nil {
			unavailable(w, "Summary is unavailable", err)
			return
		}
		var summary map[string]json.RawMessage
		if err := json.Unmarshal(content, &summary); err != nil || summary == nil {
			unavailable(w, "Summary is unavailable", errors.New("invalid prepared summary"))
			return
		}
		writeJSON(w, 200, json.RawMessage(content))
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		detail(w, http.StatusNotFound, "Not Found")
	})
	return mux
}

func detail(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"detail": message})
}

func unavailable(w http.ResponseWriter, message string, err error) {
	if !errors.Is(err, pgx.ErrNoRows) {
		log.Printf("Prophet: %s: %v", message, err)
	}
	detail(w, http.StatusServiceUnavailable, message)
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	var body bytes.Buffer
	encoder := json.NewEncoder(&body)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(data); err != nil {
		log.Printf("Prophet response encoding: %v", err)
		status = http.StatusInternalServerError
		body.Reset()
		body.WriteString(`{"detail":"Internal Server Error"}` + "\n")
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := body.WriteTo(w); err != nil {
		log.Printf("Prophet response write: %v", err)
	}
}
