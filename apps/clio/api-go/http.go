package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"log"
	"math"
	"math/big"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

var metrics = []string{
	"bx", "by", "bz", "v", "n", "t", "kp", "dst", "ap", "f10_7", "s10", "m10", "y10",
	"f10_7_81mean", "s10_81mean", "m10_81mean", "y10_81mean", "bt",
}

type observationReader interface {
	ready(ctx context.Context) error
	latest(ctx context.Context, metrics []string) ([]measurement, error)
	history(ctx context.Context, metrics []string, start, end time.Time, interval string) (numericPoints, error)
	files(ctx context.Context, observation string, start, end time.Time) ([]observationFile, error)
	sdoImages(ctx context.Context, publicBase string) ([]sdoImage, error)
	eventInputs(ctx context.Context, now time.Time) (eventInputs, error)
}

type api struct {
	db         observationReader
	publicBase string
}

type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type apiResponse struct {
	Success bool      `json:"success"`
	Data    any       `json:"data,omitempty"`
	Error   *apiError `json:"error"`
}

func newHandler(db observationReader, token, publicBase string) http.Handler {
	handler := api{db: db, publicBase: strings.TrimRight(publicBase, "/")}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health/live", func(w http.ResponseWriter, r *http.Request) {
		writeHealth(w)
	})
	mux.HandleFunc("GET /health/ready", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		if err := db.ready(ctx); err != nil {
			log.Printf("Clio readiness query: %v", err)
			writeError(w, http.StatusServiceUnavailable, "NOT_READY", "Clio storage is not ready")
			return
		}
		writeHealth(w)
	})
	for path, handler := range map[string]http.HandlerFunc{
		"/internal/v1/observations/latest":                   handler.latest,
		"/internal/v1/observations/history":                  handler.history,
		"/internal/v1/files":                                 handler.files,
		"/internal/v1/observations/sdo":                      handler.sdo,
		"/internal/v1/observations/solar-wind-events/inputs": handler.events,
	} {
		mux.HandleFunc("GET "+path, requireToken(token, handler))
	}
	return mux
}

func requireToken(token string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if token == "" {
			writeError(w, http.StatusServiceUnavailable, "NOT_READY", "Observation read service is not configured")
			return
		}
		scheme, credentials, _ := strings.Cut(r.Header.Get("Authorization"), " ")
		if !strings.EqualFold(scheme, "Bearer") || subtle.ConstantTimeCompare([]byte(credentials), []byte(token)) != 1 {
			writeError(w, http.StatusUnauthorized, "HTTP_ERROR", "Invalid service credentials")
			return
		}
		next(w, r)
	}
}

func requestQuery(r *http.Request) url.Values {
	// Python's query parser treats a raw semicolon as part of the value.
	query, _ := url.ParseQuery(strings.ReplaceAll(r.URL.RawQuery, ";", "%3B"))
	return query
}

func selectedMetrics(w http.ResponseWriter, query url.Values) []string {
	values := query["metrics"]
	if len(values) == 0 {
		return metrics
	}
	var selected []string
	for _, value := range strings.Split(values[len(values)-1], ",") {
		metric := strings.TrimSpace(value)
		if !slices.Contains(metrics, metric) {
			writeError(w, http.StatusUnprocessableEntity, "HTTP_ERROR", "Unknown observation metric")
			return nil
		}
		if !slices.Contains(selected, metric) {
			selected = append(selected, metric)
		}
	}
	return selected
}

func (api api) latest(w http.ResponseWriter, r *http.Request) {
	selected := selectedMetrics(w, requestQuery(r))
	if selected == nil {
		return
	}
	rows, err := api.db.latest(r.Context(), selected)
	if err != nil {
		log.Printf("Clio latest query: %v", err)
		writeError(w, http.StatusServiceUnavailable, "NOT_READY", "Numeric observation storage is unavailable")
		return
	}
	points := make(numericPoints)
	for _, row := range rows {
		points.set(row.observedAt, row.metric, validatedValue(row.metric, row.value))
	}
	writeSuccess(w, map[string]any{"points": points.jsonPoints(nil)})
}

func validatedValue(metric string, number *float64) *float64 {
	if number == nil || math.IsNaN(*number) || math.IsInf(*number, 0) {
		return nil
	}
	value := *number
	switch metric {
	case "s10", "m10", "y10":
		return nil
	case "bx", "by", "bz", "dst":
		// Signed observations accept negative values.
	default:
		if value < 0 || (metric == "kp" && value > 9) {
			return nil
		}
	}
	return number
}

func observationValue(metric string, number *float64) any {
	if number == nil || math.IsNaN(*number) || math.IsInf(*number, 0) {
		return nil
	}
	value := *number
	switch metric {
	case "kp", "dst", "ap", "f10_7":
		integer, _ := big.NewFloat(value).Int(nil)
		return integer
	default:
		return value
	}
}

func writeSuccess(w http.ResponseWriter, data any) {
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, apiResponse{Success: true, Data: data})
}

func writeHealth(w http.ResponseWriter) {
	writeJSON(w, http.StatusOK, map[string]string{"service": "clio", "status": "ok"})
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, apiResponse{Error: &apiError{Code: code, Message: message}})
}

func writeStorageError(w http.ResponseWriter, fallback string, err, publicError error) {
	log.Printf("Clio: %v", err)
	if publicError != nil && errors.Is(err, publicError) {
		fallback = publicError.Error()
	}
	writeError(w, http.StatusServiceUnavailable, "NOT_READY", fallback)
}

func writeJSON(w http.ResponseWriter, status int, response any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Clio response: %v", err)
	}
}
