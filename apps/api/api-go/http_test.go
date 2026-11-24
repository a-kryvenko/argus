package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type document map[string]any

func TestPublicRoutes(t *testing.T) {
	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	app, err := loadAPI(root)
	if err != nil {
		t.Fatal(err)
	}
	issue := time.Now().UTC().Truncate(time.Hour).Add(-time.Hour)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-service-token" {
			t.Error("upstream request does not have its service token")
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		switch r.URL.Path {
		case "/internal/v1/summary":
			writeJSON(w, 200, document{"observations": document{}, "generated_at": issue})
		case "/internal/v1/observations/latest", "/internal/v1/observations/history":
			if r.URL.Query().Get("metrics") == "" {
				t.Error("missing metrics selection")
			}
			success(w, document{"points": []document{
				{"time": issue.Add(-time.Hour), "bz": -3, "kp": 2, "ap": 7},
				{"time": issue, "bz": nil, "kp": 3, "ap": 13},
			}})
		case "/internal/v1/public-forecasts/solar-wind-speed":
			data := document{"target": "solar-wind-speed", "issue_time": issue, "horizon_hours": 2,
				"available_variables": []string{"v"}, "cme_speed": nil,
				"predictions": []document{{"valid_time": issue.Add(time.Hour), "lead_hours": 1,
					"variables": document{"v": document{
						"continuous": document{"q10": 380, "q50": 420, "q90": 462},
						"binary": []document{{"threshold": 450, "probability": 0.8},
							{"threshold": 500, "probability": 0.6}, {"threshold": 600, "probability": 0.4}},
					}}}}}
			if r.URL.Query().Get("meta") == "true" {
				data["meta"] = document{"variables": document{"v": document{"unit": "km/s"}}}
			}
			writeJSON(w, 200, data)
		case "/internal/v1/forecasts/dst/verification":
			writeJSON(w, 200, document{"product": "dst", "start": issue.Add(-24 * time.Hour), "end": issue, "groups": []document{}})
		case "/internal/v1/public-forecasts/solar-wind-events":
			writeJSON(w, 200, document{"schema_version": 1, "product": "solar-wind-events", "issue_time": issue,
				"model_version": "solar-wind-events-v4", "source_status": document{},
				"events": []document{}, "l1_detections": []document{}})
		case "/internal/v1/public-forecast-metrics/solar-wind-speed":
			report := document{"target": "solar-wind-speed", "variables": document{}, "evaluation": nil}
			if r.URL.Query().Get("meta") == "true" {
				report["meta"] = document{"variables": document{}}
			}
			writeJSON(w, 200, report)
		case "/internal/v1/public-forecasts/atmospheric-density":
			density := document{"target": "atmospheric-density", "predictions": make([]document, 49)}
			if r.URL.Query().Get("meta") == "true" {
				density["meta"] = document{"model": "JB2008"}
			}
			writeJSON(w, 200, density)
		case "/internal/v1/public-risks/leo-drag":
			var input LeoDragRequest
			if err := json.NewDecoder(r.Body).Decode(&input); err != nil || (input.Horizon != 24 && input.Horizon != 48) {
				t.Errorf("unexpected upstream drag request: %v %+v", err, input)
				w.WriteHeader(400)
				return
			}
			if input.Horizon == 48 {
				writeJSON(w, 422, document{"detail": "private impact service detail"})
				return
			}
			writeJSON(w, 200, dragAssessmentFixture(input, issue, r.URL.Query().Get("meta") == "true"))
		default:
			w.WriteHeader(503)
		}
	}))
	defer upstream.Close()
	app.clio = configuredTestUpstream(upstream)
	app.prophet = app.clio
	app.intelligence = app.clio
	handler := app.routes()

	check := func(method, path, body string, status int) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(method, prefix+path, strings.NewReader(body))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		if response.Code != status {
			t.Fatalf("%s %s = %d, want %d: %s", method, path, response.Code, status, response.Body)
		}
		return response
	}
	t.Run("documentation", func(t *testing.T) {
		got := check("GET", "/openapi.json", "", 200)
		var specification struct {
			OpenAPI string `json:"openapi"`
			Paths   map[string]map[string]struct {
				Responses map[string]json.RawMessage `json:"responses"`
			} `json:"paths"`
		}
		if err := json.Unmarshal(got.Body.Bytes(), &specification); err != nil || !strings.HasPrefix(specification.OpenAPI, "3.1.") {
			t.Fatalf("invalid OpenAPI response: %v", err)
		}
		want := []string{"/ping", "/observations", "/observations/history", "/summary",
			"/forecasts/solar-wind-events", "/forecasts/atmospheric-density", "/forecasts/{target}",
			"/forecasts/{target}/metrics", "/forecasts/{target}/verification", "/risks/leo-drag"}
		for _, path := range want {
			method := "get"
			if path == "/risks/leo-drag" {
				method = "post"
			}
			responses := specification.Paths[path][method].Responses
			if responses["200"] == nil || (path != "/ping" && responses["503"] == nil) {
				t.Errorf("missing response definition for %s", path)
			}
		}
		if got := check("GET", "/docs", "", 200); !strings.Contains(got.Body.String(), prefix+"/openapi.json") {
			t.Fatal("Swagger UI does not point to the public schema")
		}
		check("GET", "/dashboard/me", "", 404)
	})
	t.Run("read and error envelopes", func(t *testing.T) {
		if got := responseData[string](t, check("GET", "/ping", "", 200)); got != "pong" {
			t.Fatalf("ping = %s", got)
		}
		if got := check("GET", "/summary", "", 200); got.Header().Get("Cache-Control") != "no-store" {
			t.Fatal("summary must not be cached")
		}
		got := check("GET", "/forecasts/unknown", "", 404)
		if !strings.Contains(got.Body.String(), `"code":"NOT_FOUND"`) {
			t.Fatal(got.Body.String())
		}
		check("GET", "/forecasts/hmf", "", 503)
		check("GET", "/observations?meta=invalid", "", 422)
		check("GET", "/observations/history?observations=bz&from=2026-11-24T00:00:00", "", 422)
		check("POST", "/risks/leo-drag", `{"altitude_km":12}`, 422)
		check("POST", "/risks/leo-drag", strings.Repeat(" ", (1<<16)+1), 413)
	})
	t.Run("observations", func(t *testing.T) {
		current := responseData[map[string]json.RawMessage](t, check("GET", "/observations?observations=bz,kp,ap&meta=true&trends=true", "", 200))
		if string(current["meta"]) == "" || string(current["bz"]) == "null" ||
			!strings.Contains(string(current["bz"]), `"value":null`) ||
			strings.Contains(string(current["kp"]), "interval_status") {
			t.Fatalf("invalid compact observations: %s", current)
		}
		query := "/observations/history?observations=bz,kp,ap&resolution=auto&meta=true&from=" +
			issue.Add(-2*time.Hour).Format(time.RFC3339) + "&to=" + issue.Add(time.Hour).Format(time.RFC3339)
		history := responseData[map[string]json.RawMessage](t, check("GET", query, "", 200))
		if string(history["resolution_seconds"]) != "60" || !strings.Contains(string(history["series"]), `"invalid_slots":1`) ||
			!strings.Contains(string(history["series"]), `"basis":"overlapping_intervals"`) ||
			!strings.Contains(string(history["points"]), `"ap":13`) {
			t.Fatalf("invalid observation history: %s", history)
		}
	})
	t.Run("forecasts", func(t *testing.T) {
		compact := responseData[document](t, check("GET", "/forecasts/solar-wind-speed", "", 200))
		expanded := responseData[document](t, check("GET", "/forecasts/solar-wind-speed?meta=true", "", 200))
		if expanded["meta"] == nil {
			t.Fatal("missing forecast metadata")
		}
		delete(expanded, "meta")
		if !reflect.DeepEqual(compact, expanded) || compact["horizon_hours"] != float64(2) {
			t.Fatalf("forecast data differs with metadata: %v", expanded)
		}
		points := compact["predictions"].([]any)
		speed := points[0].(map[string]any)["variables"].(map[string]any)["v"].(map[string]any)
		if speed["continuous"].(map[string]any)["q90"] != float64(462) || len(speed["binary"].([]any)) != 3 {
			t.Fatalf("invalid forecast point: %v", speed)
		}
		metrics := responseData[document](t, check("GET", "/forecasts/solar-wind-speed/metrics?meta=true", "", 200))
		if metrics["meta"] == nil {
			t.Fatal("metrics metadata was not forwarded")
		}
		check("GET", "/forecasts/hmf/metrics", "", 503)
		check("GET", "/forecasts/dst/verification", "", 200)
		check("GET", "/forecasts/solar-wind-events", "", 200)
		density := responseData[document](t, check("GET", "/forecasts/atmospheric-density?meta=true", "", 200))
		if len(density["predictions"].([]any)) != 49 || density["meta"] == nil {
			t.Fatalf("invalid density forecast: %v", density)
		}
	})
	t.Run("drag", func(t *testing.T) {
		const body = `{"altitude_km":400,"inclination_deg":51,"mass_kg":100,"effective_area_m2":1,"drag_coefficient":2}`
		input := strings.TrimSuffix(body, "}")
		failure := check("POST", "/risks/leo-drag", input+`,"horizon_hours":48}`, 422)
		if strings.Contains(failure.Body.String(), "private impact service detail") {
			t.Fatal("upstream error details reached the public API")
		}
		compact := responseData[document](t, check("POST", "/risks/leo-drag", body, 200))
		if compact["meta"] != nil {
			t.Fatal("metadata was included for the compact response")
		}
		check("POST", "/risks/leo-drag", input+`,"horizon_hours":null}`, 422)
		check("POST", "/risks/leo-drag", input+`,"thresholds":{"elevated_altitude_loss_m":0,"high_altitude_loss_m":3}}`, 422)
		check("POST", "/risks/leo-drag", input+`,"thresholds":{"high_altitude_loss_m":3}}`, 422)
		got := responseData[document](t, check("POST", "/risks/leo-drag?meta=true", input+`,"thresholds":null}`, 200))
		if got["model"] != nil || got["meta"] == nil || len(got["predictions"].([]any)) != 25 {
			t.Fatalf("invalid drag projection: %v", got)
		}
	})
	t.Run("rollout smoke", func(t *testing.T) {
		server := httptest.NewServer(handler)
		defer server.Close()
		if err := smoke(server.URL, ""); err != nil {
			t.Fatal(err)
		}
	})
}

func configuredTestUpstream(server *httptest.Server) upstream {
	return upstream{server.URL, "test-service-token"}
}

func responseData[T any](t *testing.T, response *httptest.ResponseRecorder) T {
	t.Helper()
	var result struct {
		Success bool `json:"success"`
		Data    T    `json:"data"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || !result.Success {
		t.Fatalf("invalid public response: %v %s", err, response.Body)
	}
	return result.Data
}

func dragAssessmentFixture(input LeoDragRequest, issue time.Time, meta bool) document {
	points := make([]document, input.Horizon+1)
	for lead := range points {
		points[lead] = document{"valid_time": issue.Add(time.Duration(lead) * time.Hour), "lead_hours": lead,
			"mean_density_kg_m3": 1e-12, "mean_drag_accel_m_s2": 0.01, "mean_along_track_deceleration_m_s2": 0.01,
			"estimated_altitude_loss_m": lead, "delta_v_loss_m_s": lead}
	}
	data := document{"inputs": input, "computed_at": issue, "start_time": issue,
		"end_time": issue.Add(time.Duration(input.Horizon) * time.Hour), "mean_density_kg_m3": 1e-12,
		"mean_drag_accel_m_s2": 0.01, "estimated_altitude_loss_m": input.Horizon, "delta_v_loss_m_s": input.Horizon,
		"drag_risk": "not_assessed", "risk_reason": "No thresholds", "assumptions": []string{}, "predictions": points,
		"source": document{"issue_time": issue, "observed_at": issue, "dtc_observed_at": issue,
			"geomagnetic_observed_at": issue, "driver_mode": "observed_persistence", "background_interpolated": false,
			"release_id": "b2be2c45-bcf3-431f-b676-0c98b99b82fa"}}
	if meta {
		data["meta"] = document{"model": "circular_leo_drag_v1", "source": document{"model": "JB2008"}}
	}
	return data
}

func TestStartupConfigAndSchema(t *testing.T) {
	root := t.TempDir()
	configs := filepath.Join(root, "configs")
	if err := os.Mkdir(configs, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(configs, "project.yaml"), []byte("clio: {}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadAPI(root); err != nil {
		t.Fatalf("gateway requires forecast configuration: %v", err)
	}
}
