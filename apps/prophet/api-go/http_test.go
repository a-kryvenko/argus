package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
	"github.com/jackc/pgx/v5"
)

type emptyStore struct{}

func (emptyStore) ready(context.Context) error { return nil }
func (emptyStore) readRelease(context.Context, string, string) (release, error) {
	return release{}, pgx.ErrNoRows
}
func (emptyStore) status(context.Context, time.Time, float64) ([]forecastStatus, error) {
	return []forecastStatus{}, nil
}
func (emptyStore) solarIndices(context.Context) (solarIndices, error) {
	return solarIndices{}, pgx.ErrNoRows
}
func (emptyStore) verification(context.Context, string) (json.RawMessage, error) {
	return json.RawMessage(`{"product":"dst","groups":[]}`), nil
}

func TestRoutes(t *testing.T) {
	summary := filepath.Join(t.TempDir(), "latest.json")
	if err := os.WriteFile(summary, []byte(`{"observations":{}}`), 0600); err != nil {
		t.Fatal(err)
	}
	handler := newHandler(emptyStore{}, "token", summary, 6, nil, metricReports{})
	const prefix = "/internal/v1/forecasts/"
	const releaseID = "b2be2c45-bcf3-431f-b676-0c98b99b82fa"
	for _, tc := range []struct {
		path, token string
		code        int
	}{
		{"/health/ready", "", 200},
		{prefix + "dst/latest", "", 401},
		{prefix + "dst/latest", "wrong", 401},
		{prefix + "dst/latest", "token", 503},
		{prefix + "bad/latest", "token", 404},
		{prefix + "dst/releases/" + releaseID, "token", 404},
		{prefix + "dst/releases/b2be2c45Xbcf3X431fXb676X0c98b99b82fa", "token", 422},
		{prefix + "dst/releases/invalid", "token", 422},
		{prefix + "dst/verification", "token", 200},
		{prefix + "solar-wind-events/verification", "token", 404},
		{prefix + "status", "token", 200},
		{"/internal/v1/solar-indices", "token", 503},
		{"/internal/v1/summary", "token", 200},
	} {
		t.Run(tc.path+"/"+tc.token, func(t *testing.T) {
			request := httptest.NewRequest("GET", tc.path, nil)
			if tc.token != "" {
				request.Header.Set("Authorization", "Bearer "+tc.token)
			}
			result := httptest.NewRecorder()
			handler.ServeHTTP(result, request)
			if result.Code != tc.code || !json.Valid(result.Body.Bytes()) {
				t.Fatalf("got %d %s, expected %d and JSON", result.Code, result.Body, tc.code)
			}
		})
	}
}

func TestGeomagneticMetricsFromSharedModel(t *testing.T) {
	workdir := t.TempDir()
	configs := filepath.Join(workdir, "configs")
	if err := os.Mkdir(configs, 0700); err != nil {
		t.Fatal(err)
	}
	const registry = "models:\n  geomagnetic_ordinal:\n    metrics_report: configs/geomagnetic-activity.json\n" +
		"  geomagnetic_fast_l1:\n    metrics_report: configs/geomagnetic-fast-l1.json\n"
	if err := os.WriteFile(filepath.Join(configs, "models_registry.yaml"), []byte(registry), 0600); err != nil {
		t.Fatal(err)
	}
	const evaluation = `{
		"target": "geomagnetic-activity",
		"variables": {
			"kp": {"binary": [
				{"threshold": 4, "by_lead_hour": [{"lead_hours": 48, "brier_score": 0.12}]},
				{"threshold": 5, "by_lead_hour": [{"lead_hours": 48, "brier_score": 0.08}]},
				{"threshold": 6, "by_lead_hour": [{"lead_hours": 48, "brier_score": 0.04}]}
			]},
			"ap": {"continuous": {"quantiles": [0.1, 0.5, 0.9], "by_lead_hour": [
				{"lead_hours": 48, "values": {"mae": 5.1}}
			]}}
		}
	}`
	if err := os.WriteFile(filepath.Join(configs, "geomagnetic-activity.json"), []byte(evaluation), 0600); err != nil {
		t.Fatal(err)
	}
	models, err := loadModelRegistry(workdir)
	if err != nil {
		t.Fatal(err)
	}
	handler := newHandler(emptyStore{}, "token", "", 6, nil, metricReports{workdir, models})
	request := httptest.NewRequest(http.MethodGet, metricsPath+"geomagnetic-activity", nil)
	request.Header.Set("Authorization", "Bearer token")
	result := httptest.NewRecorder()
	handler.ServeHTTP(result, request)
	if result.Code != http.StatusOK {
		t.Fatalf("got %d %s", result.Code, result.Body)
	}
	var report publicforecast.ForecastMetrics
	if err := json.Unmarshal(result.Body.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	kp, ap := report.Variables["kp"], report.Variables["ap"]
	if report.Target != "geomagnetic-activity" || len(kp.Binary) != 3 || ap.Continuous == nil ||
		len(ap.Continuous.ByLeadHour) != 1 || ap.Continuous.ByLeadHour[0].Values["mae"] != 5.1 {
		t.Fatalf("missing Kp/ap metrics: %+v", report)
	}
	for i, threshold := range []float64{4, 5, 6} {
		series := kp.Binary[i]
		if series.Threshold == nil || *series.Threshold != threshold || len(series.ByLeadHour) != 1 ||
			series.ByLeadHour[0].BrierScore == nil {
			t.Errorf("missing scores for Kp >= %.0f: %+v", threshold, series)
		}
	}
	if report.FastL1Evaluation != nil {
		t.Fatal("the hourly metrics must be available without a fast L1 report")
	}

	const fast = `{
		"period": "January–June 2026",
		"sample": "Matched native three-hour intervals",
		"notes": ["Simulated delivery delay"],
		"by_delay": [{
			"delay_minutes": 15, "targets": 58, "updates": 55, "observed_events": 10,
			"baseline": {"kp4_brier": 0.176, "kp5_brier": 0.141, "kp6_brier": 0.083, "ap_mae": 20.5,
				"ap_coverage_80": 0.621, "ap_interval_width_80": 18.3, "alert_threshold": 0.136, "hits": 4, "false_alerts": 0},
			"candidate": {"kp4_brier": 0.142, "kp5_brier": 0.114, "kp6_brier": 0.067, "ap_mae": 17.0,
				"ap_coverage_80": 0.897, "ap_interval_width_80": 58.8, "alert_threshold": 0.361, "hits": 4, "false_alerts": 1},
			"ap_mae_delta_ci95": [-5.52, -0.86], "kp5_brier_delta_ci95": [-0.074, 0.029]
		}]
	}`
	fastFile := filepath.Join(configs, "geomagnetic-fast-l1.json")
	for _, tc := range []struct {
		name      string
		content   string
		available bool
	}{
		{"paired metrics", fast, true},
		{"invalid target coverage", strings.Replace(fast, `"updates": 55`, `"updates": 59`, 1), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := os.WriteFile(fastFile, []byte(tc.content), 0600); err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			var parsed publicforecast.ForecastMetrics
			if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &parsed) != nil ||
				parsed.Variables["ap"].Continuous.ByLeadHour[0].Values["mae"] != 5.1 {
				t.Fatalf("missing hourly metrics: %d %s", response.Code, response.Body)
			}
			if (parsed.FastL1Evaluation != nil) != tc.available {
				t.Fatalf("wrong fast report availability: %s", response.Body)
			}
			if tc.available {
				row := parsed.FastL1Evaluation.ByDelay[0]
				if row.DelayMinutes != 15 || row.Targets != 58 || row.Updates != 55 || row.Candidate.Kp5Brier != .114 ||
					row.Baseline.ApMAE != 20.5 || row.Candidate.FalseAlerts != 1 {
					t.Fatalf("incomplete paired comparison: %+v", row)
				}
			}
		})
	}
}

type unavailableStore struct{ emptyStore }

var errPrivateStorage = errors.New("private storage details")

func (unavailableStore) readRelease(context.Context, string, string) (release, error) {
	return release{}, errPrivateStorage
}

func (unavailableStore) status(context.Context, time.Time, float64) ([]forecastStatus, error) {
	return nil, errPrivateStorage
}

func (unavailableStore) solarIndices(context.Context) (solarIndices, error) {
	return solarIndices{}, errPrivateStorage
}

func TestProtectedRoutesReturnPublicErrors(t *testing.T) {
	summary := filepath.Join(t.TempDir(), "missing.json")
	for _, handler := range []http.Handler{
		newHandler(unavailableStore{}, "token", summary, 6, nil, metricReports{}),
		newHandler(emptyStore{}, "", summary, 6, nil, metricReports{}),
	} {
		for _, path := range []string{
			"/internal/v1/forecasts/dst/latest",
			"/internal/v1/forecasts/status",
			"/internal/v1/solar-indices",
			"/internal/v1/summary",
		} {
			request := httptest.NewRequest(http.MethodGet, path, nil)
			request.Header.Set("Authorization", "Bearer token")
			result := httptest.NewRecorder()
			handler.ServeHTTP(result, request)
			if result.Code != http.StatusServiceUnavailable ||
				strings.Contains(result.Body.String(), errPrivateStorage.Error()) ||
				strings.Contains(result.Body.String(), summary) {
				t.Errorf("%s returned %d %s", path, result.Code, result.Body)
			}
		}
	}
}

type statusDeadlineStore struct {
	emptyStore
	t *testing.T
}

func (store statusDeadlineStore) status(ctx context.Context, assessedAt time.Time, _ float64) ([]forecastStatus, error) {
	deadline, ok := ctx.Deadline()
	if !ok || !deadline.After(assessedAt) || deadline.After(assessedAt.Add(statusTimeout)) {
		store.t.Errorf("status deadline = %v (present: %t), assessed at %v", deadline, ok, assessedAt)
	}
	return nil, context.DeadlineExceeded
}

func TestStatusHasDeadline(t *testing.T) {
	handler := newHandler(statusDeadlineStore{t: t}, "token", "", 6, nil, metricReports{})
	request := httptest.NewRequest(http.MethodGet, "/internal/v1/forecasts/status", nil)
	request.Header.Set("Authorization", "Bearer token")
	result := httptest.NewRecorder()
	handler.ServeHTTP(result, request)
	if result.Code != http.StatusServiceUnavailable || !json.Valid(result.Body.Bytes()) {
		t.Fatalf("got %d %s", result.Code, result.Body)
	}
}

func TestWriteJSONReturnsInternalErrorWhenEncodingFails(t *testing.T) {
	for _, data := range []any{math.NaN(), json.RawMessage(`{"invalid":`)} {
		result := httptest.NewRecorder()
		writeJSON(result, http.StatusOK, data)
		if result.Code != http.StatusInternalServerError ||
			result.Header().Get("Content-Type") != "application/json" ||
			result.Body.String() != "{\"detail\":\"Internal Server Error\"}\n" {
			t.Errorf("got %d %s", result.Code, result.Body)
		}
	}
}

func TestDensityMaxAgeRejectsNonFiniteValues(t *testing.T) {
	dir := t.TempDir()
	configs := filepath.Join(dir, "configs")
	if err := os.Mkdir(configs, 0700); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{".nan", ".inf", "-.inf"} {
		registry := "models:\n  atmospheric_density:\n    max_age_hours: " + value + "\n"
		if err := os.WriteFile(filepath.Join(configs, "models_registry.yaml"), []byte(registry), 0600); err != nil {
			t.Fatal(err)
		}
		models, err := loadModelRegistry(dir)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := models.densityMaxAge(); err == nil || !strings.Contains(err.Error(), "must be finite") {
			t.Errorf("max_age_hours = %s returned %v", value, err)
		}
	}
}

func TestReleaseFreshness(t *testing.T) {
	limit := 6.0
	for _, test := range []struct {
		age    float64
		maxAge *float64
		want   string
	}{
		{6, &limit, "within_age_limit"},
		{6 + 1.0/3600, &limit, "stale"},
		{30 * 24, nil, "unconfigured"},
		{-1.0 / 3600, nil, "future_issue_time"},
	} {
		if got := releaseFreshness(test.age, test.maxAge); got != test.want {
			t.Errorf("releaseFreshness(%v, %v) = %q, want %q", test.age, test.maxAge, got, test.want)
		}
	}
}

func TestArtifactDecodePreservesBytesAndChecksHash(t *testing.T) {
	const content = "label,value\r\n\"é,hi\",3\r\n"
	var buffer bytes.Buffer
	compress := gzip.NewWriter(&buffer)
	if _, err := compress.Write([]byte(content)); err != nil {
		t.Fatal(err)
	}
	if err := compress.Close(); err != nil {
		t.Fatal(err)
	}
	item := artifact{
		SHA256:  fmt.Sprintf("%x", sha256.Sum256([]byte(content))),
		Columns: []string{"label", "value"}, RowCount: 1,
		ModelInfo: json.RawMessage(`{}`),
	}
	if err := item.decode(buffer.Bytes()); err != nil || item.CSVText != content {
		t.Fatalf("unexpected artifact: %q, %v", item.CSVText, err)
	}
	item.SHA256 = "wrong"
	if err := item.decode(buffer.Bytes()); err == nil {
		t.Fatal("accepted an invalid checksum")
	}
}
