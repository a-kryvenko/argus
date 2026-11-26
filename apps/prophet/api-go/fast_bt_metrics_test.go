package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

func TestFastBtMetricsWithoutHourlyReport(t *testing.T) {
	workdir := t.TempDir()
	configs := filepath.Join(workdir, "configs")
	if err := os.Mkdir(configs, 0700); err != nil {
		t.Fatal(err)
	}
	const registry = "models:\n  hmf_fast_l1_bt:\n    metrics_report: configs/hmf-fast-l1.json\n"
	if err := os.WriteFile(filepath.Join(configs, "models_registry.yaml"), []byte(registry), 0600); err != nil {
		t.Fatal(err)
	}
	models, err := loadModelRegistry(workdir)
	if err != nil {
		t.Fatal(err)
	}
	handler := newHandler(emptyStore{}, "token", "", 6, nil, metricReports{workdir, models})
	request := httptest.NewRequest(http.MethodGet, metricsPath+"hmf", nil)
	request.Header.Set("Authorization", "Bearer token")
	const scores = `{"brier_score":0.1,"average_precision":null,"alert_threshold":0.8,"hits":0,"false_alerts":1}`
	const report = `{"period":"January–June 2026","sample":"Complete native L1 hours","notes":[],"cohorts":[{
		"id":"all_triggers","label":"All L1 detections","by_delay":[{"delay_minutes":15,"targets":100,"updates":99,"thresholds":[
			{"threshold":5,"observed_events":82,"baseline":` + scores + `,"candidate":` + scores + `},
			{"threshold":10,"observed_events":42,"baseline":` + scores + `,"candidate":` + scores + `,"brier_delta_ci95":[-0.3,0.1]},
			{"threshold":15,"observed_events":0,"baseline":` + scores + `,"candidate":` + scores + `}
		]}]
	}]}`
	for _, tc := range []struct {
		name   string
		body   string
		status int
	}{
		{"archive without hourly report", report, http.StatusOK},
		{"impossible update count", strings.Replace(report, `"updates":99`, `"updates":101`, 1), http.StatusServiceUnavailable},
		{"missing probability score", strings.Replace(report, `"brier_score":0.1,`, "", 1), http.StatusServiceUnavailable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := os.WriteFile(filepath.Join(configs, "hmf-fast-l1.json"), []byte(tc.body), 0600); err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != tc.status {
				t.Fatalf("got %d %s, expected %d", response.Code, response.Body, tc.status)
			}
			if tc.status != http.StatusOK {
				return
			}
			var data publicforecast.ForecastMetrics
			if err := json.Unmarshal(response.Body.Bytes(), &data); err != nil {
				t.Fatal(err)
			}
			if data.Target != "hmf" || data.FastBtEvaluation == nil || data.Variables == nil || len(data.Variables) != 0 || data.Meta == nil {
				t.Fatalf("expected a Bt archive report: %s", response.Body)
			}
			row := data.FastBtEvaluation.Cohorts[0].ByDelay[0]
			threshold := row.Thresholds[1]
			if row.DelayMinutes != 15 || row.Targets != 100 || row.Updates != 99 || threshold.Threshold != 10 ||
				threshold.ObservedEvents != 42 || threshold.Candidate.AveragePrecision != nil ||
				threshold.Candidate.BrierScore == nil || *threshold.Candidate.BrierScore != 0.1 {
				t.Fatalf("incomplete paired comparison: %+v", row)
			}
		})
	}
}
