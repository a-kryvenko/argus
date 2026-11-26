package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
	"github.com/a-kryvenko/argus/observationcatalog"
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
		case "/internal/v1/observations/series":
			query := r.URL.Query()
			data := make(map[string]observationcatalog.Series)
			add := func(at time.Time, value *float64, product string) observationcatalog.Point {
				return observationcatalog.Point{ObservedAt: at, Value: value, SourceProduct: product}
			}
			for _, id := range strings.Split(query.Get("metrics"), ",") {
				metric, found := observationcatalog.Find(id)
				if !found {
					t.Errorf("invalid metric in Clio request: %s", id)
					continue
				}
				series := observationcatalog.Series{ResolutionSeconds: metric.NativeSeconds,
					TimeBasis: "measurement", Aggregation: "source", Points: []observationcatalog.Point{}}
				if metric.SourceInterval {
					series.TimeBasis = "source_interval_start"
				}
				if metric.Group == "wind" {
					series.TimeBasis = l1TimeBasis
				}
				switch id {
				case "bz":
					series.TimeBasis = l1TimeBasis
					series.Points = []observationcatalog.Point{
						add(issue.Add(-time.Hour), pointer(-3.0), "swpc.rtsw_mag"),
						add(issue, nil, "swpc.rtsw_mag"),
					}
					series.Points[1].Quality = "flagged"
					if query.Get("interval") == "1h" {
						series.Aggregation, series.ResolutionSeconds = "mean", 3600
						series.Points = series.Points[:1]
						series.Points[0].Count, series.Points[0].ExpectedCount = pointer(12), pointer(60)
					}
					if query.Get("time_basis") == nearEarthTimeBasis {
						if query.Get("interval") != "1h" {
							t.Errorf("Near-Earth wind resolution: %v", query)
						}
						series.TimeBasis, series.Aggregation = nearEarthTimeBasis, "source"
						series.Points = []observationcatalog.Point{add(issue, pointer(-8.25), "omni.hourly")}
					}
					if !query.Has("from") {
						series.Points = series.Points[len(series.Points)-1:]
					}
				case "n":
					series.TimeBasis = l1TimeBasis
					if query.Get("metrics") == "n" {
						series.TimeBasis = "invalid_basis"
					}
				case "kp":
					series.TimeBasis = "source_interval_start"
					series.Points = []observationcatalog.Point{add(issue.Truncate(3*time.Hour), pointer(2.67), "swpc.kp")}
				case "ap":
					series.Points = []observationcatalog.Point{add(issue.Truncate(3*time.Hour), pointer(13.0), "swpc.kp")}
				case "xrs_a":
					series.Points = []observationcatalog.Point{add(issue.Add(time.Minute), pointer(1.7e-8), "goes.xrs")}
				case "xrs_b":
					series.Points = []observationcatalog.Point{add(issue.Add(time.Minute), pointer(8.5e-7), "goes.xrs")}
				case "proton_10mev":
					series.Points = []observationcatalog.Point{add(issue.Add(5*time.Minute), pointer(2.5), "goes.protons")}
				case "proton_100mev":
					series.Points = []observationcatalog.Point{add(issue.Add(5*time.Minute), pointer(0.0125), "goes.protons")}
				}
				data[id] = series
			}
			success(w, data)
		case "/internal/v1/public-forecasts/solar-wind-speed":
			data := document{"target": "solar-wind-speed", "issue_time": issue, "horizon_hours": 1,
				"available_variables": []string{"v"}, "cme_speed": nil,
				"predictions": []document{{"valid_time": issue.Add(time.Hour), "lead_hours": 1,
					"variables": document{"v": document{
						"continuous": document{"q10": 380, "q50": 420, "q90": 462},
						"binary": []document{{"threshold": 450, "probability": 0.8},
							{"threshold": 500, "probability": 0.6}, {"threshold": 600, "probability": 0.4}},
					}}}}}
			data["meta"] = document{"variables": document{"v": document{"unit": "km/s"}}}
			writeJSON(w, 200, data)
		case "/internal/v1/public-forecasts/geomagnetic-activity":
			target := issue.Truncate(3 * time.Hour).Add(3 * time.Hour)
			lead := int(target.Sub(issue) / time.Hour)
			writeJSON(w, 200, publicforecast.Forecast{
				Target: "geomagnetic-activity", IssueTime: issue, HorizonHours: lead,
				AvailableVariables: []string{"kp", "ap"}, Predictions: []*publicforecast.ForecastPoint{{
					Valid: target, Lead: lead, UpdateIssuedAt: pointer(target.Add(-49*time.Minute - 3*time.Second)),
					Variables: map[string]*publicforecast.VariableForecast{
						"kp": {Binary: []publicforecast.BinaryForecast{{Threshold: 5, Probability: .65}}},
						"ap": {Continuous: &publicforecast.QuantileForecast{Q10: 14, Q50: 48, Q90: 94}},
					},
				}},
			})
		case "/internal/v1/public-forecasts/hmf":
			target := issue.Add(time.Hour)
			writeJSON(w, 200, publicforecast.Forecast{
				Target: "hmf", IssueTime: issue, HorizonHours: 1,
				AvailableVariables: []string{"bt", "bs"}, Predictions: []*publicforecast.ForecastPoint{{
					Valid: target, Lead: 1,
					Variables: map[string]*publicforecast.VariableForecast{
						"bt": {Binary: []publicforecast.BinaryForecast{{Threshold: 10, Probability: .8}},
							UpdateIssuedAt: pointer(target.Add(-49*time.Minute - 3*time.Second))},
						"bs": {Binary: []publicforecast.BinaryForecast{{Threshold: 10, Probability: .1}}},
					},
				}},
			})
		case "/internal/v1/forecasts/dst/verification":
			writeJSON(w, 200, document{"product": "dst", "start": issue.Add(-24 * time.Hour), "end": issue, "groups": []document{}})
		case "/internal/v1/public-forecasts/solar-wind-events":
			writeJSON(w, 200, document{"schema_version": 1, "product": "solar-wind-events", "issue_time": issue,
				"model_version": "solar-wind-events-v4", "source_status": document{},
				"events": []document{}, "l1_detections": []document{}})
		case "/internal/v1/public-forecast-metrics/solar-wind-speed":
			report := document{"target": "solar-wind-speed", "variables": document{}, "evaluation": nil}
			report["meta"] = document{"variables": document{}}
			writeJSON(w, 200, report)
		case "/internal/v1/public-forecasts/atmospheric-density":
			points := make([]document, 49)
			for lead := range points {
				points[lead] = document{"lead_hours": lead, "valid_time": issue.Add(time.Duration(lead) * time.Hour),
					"cells": []document{{"altitude_km": 400, "latitude_deg": 0,
						"rho_kg_m3": 1e-12, "rho_lon_p10_kg_m3": 1e-13, "rho_lon_p90_kg_m3": 1e-11}}}
			}
			density := document{"target": "atmospheric-density", "driver_mode": "observed_persistence",
				"issue_time": issue, "observed_at": issue, "horizon_hours": 48, "predictions": points}
			density["meta"] = document{"model": "JB2008"}
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
			writeJSON(w, 200, dragAssessmentFixture(input, issue))
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
	t.Run("versioned health routes", func(t *testing.T) {
		for _, path := range []string{"/health/live", "/health/ready"} {
			check("GET", path, "", 200)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
			if response.Code != http.StatusNotFound {
				t.Errorf("GET %s = %d, want 404", path, response.Code)
			}
		}
	})
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
		want := []string{"/ping", "/observations", "/observations/history", "/observations/catalog", "/summary",
			"/forecasts/solar-wind-events", "/forecasts/atmospheric-density", "/forecasts/{target}",
			"/forecasts/{target}/metrics", "/forecasts/{target}/verification", "/risks/leo-drag"}
		for _, path := range want {
			method := "get"
			if path == "/risks/leo-drag" {
				method = "post"
			}
			responses := specification.Paths[path][method].Responses
			if responses["200"] == nil || (path != "/ping" && path != "/observations/catalog" && responses["503"] == nil) {
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
		check("GET", "/forecasts/solar-radiation", "", 503)
		check("GET", "/observations/history?observations=bz&from=2026-11-24T00:00:00", "", 422)
		check("POST", "/risks/leo-drag", `{"altitude_km":12}`, 422)
		check("POST", "/risks/leo-drag", strings.Repeat(" ", (1<<16)+1), 413)
	})
	t.Run("observations", func(t *testing.T) {
		current := responseData[map[string]json.RawMessage](t, check("GET", "/observations?observations=bz,kp,ap&trends=true", "", 200))
		if current["meta"] != nil || string(current["bz"]) == "null" ||
			!strings.Contains(string(current["bz"]), `"value":null`) ||
			!strings.Contains(string(current["kp"]), `"value":2.67`) ||
			strings.Contains(string(current["kp"]), "interval_status") {
			t.Fatalf("invalid compact observations: %s", current)
		}
		query := "/observations/history?observations=bz,kp,ap&resolution=auto&from=" +
			issue.Add(-2*time.Hour).Format(time.RFC3339) + "&to=" + issue.Add(time.Hour).Format(time.RFC3339)
		history := responseData[map[string]json.RawMessage](t, check("GET", query, "", 200))
		if string(history["time_basis"]) != `"l1"` || string(history["resolution_seconds"]) != "60" ||
			!strings.Contains(string(history["series"]), `"invalid_slots":1`) ||
			!strings.Contains(string(history["series"]), `"basis":"overlapping_intervals"`) ||
			!strings.Contains(string(history["points"]), `"ap":13`) {
			t.Fatalf("invalid observation history: %s", history)
		}
	})
	t.Run("wind time basis", func(t *testing.T) {
		bounds := "&from=" + issue.Add(-time.Hour).Format(time.RFC3339) + "&to=" + issue.Add(30*time.Minute).Format(time.RFC3339)
		const path = "/observations/history?observations=bz"
		near := responseData[ObservationHistory](t, check("GET", path+"&time_basis=near_earth&resolution=auto"+bounds, "", 200))
		if near.TimeBasis == nil || *near.TimeBasis != nearEarthTimeBasis {
			t.Fatalf("Near-Earth wind basis: %v", near.TimeBasis)
		}
		l1 := responseData[struct {
			Series map[string]SolarHistorySeries `json:"series"`
		}](t, check("GET", path+"&resolution=1h"+bounds, "", 200))
		series := l1.Series["bz"]
		if series.Coverage.ExpectedSlots != 1 || len(series.Points) != 1 || series.Points[0].Count == nil || *series.Points[0].Count != 12 ||
			series.Points[0].SourceProduct != "swpc.rtsw_mag" {
			t.Fatalf("L1 hourly coverage: %+v", l1)
		}
		check("GET", path+"&time_basis=near_earth&resolution=5m"+bounds, "", 422)
		check("GET", path+"&time_basis="+bounds, "", 422)
		check("GET", "/observations/history?observations=n", "", 503)
	})
	t.Run("GOES XRS and proton flux", func(t *testing.T) {
		current := responseData[CurrentObservations](t, check("GET", "/observations?observations=xrs_b,proton_10mev", "", 200))
		if current.XRSB == nil || *current.XRSB == nil || *(**current.XRSB).Value != 8.5e-7 ||
			current.Proton10MeV == nil || *current.Proton10MeV == nil || *(**current.Proton10MeV).Value != 2.5 {
			t.Fatalf("invalid current GOES data: %+v", current)
		}
		const path = "/observations/history?observations=xrs_a,xrs_b,proton_10mev,proton_100mev&resolution=1h"
		bounds := "&from=" + issue.Format(time.RFC3339) + "&to=" + issue.Add(10*time.Minute).Format(time.RFC3339)
		history := responseData[struct {
			Series map[string]SolarHistorySeries `json:"series"`
		}](t, check("GET", path+bounds, "", 200))
		for _, tc := range []struct {
			metric string
			value  float64
			step   int
		}{{"xrs_a", 1.7e-8, 60}, {"xrs_b", 8.5e-7, 60}, {"proton_10mev", 2.5, 300}, {"proton_100mev", 0.0125, 300}} {
			series := history.Series[tc.metric]
			if len(series.Points) != 1 || series.Points[0].Value == nil || *series.Points[0].Value != tc.value ||
				!series.Points[0].ObservedAt.Equal(issue.Add(time.Duration(tc.step)*time.Second)) ||
				series.Coverage.ResolutionSeconds != tc.step || series.Coverage.Basis != sampledCoverage {
				t.Fatalf("%s history: %+v", tc.metric, series)
			}
		}
		check("GET", path+"&from="+issue.Add(-8*24*time.Hour).Format(time.RFC3339), "", 422)
	})
	t.Run("forecasts", func(t *testing.T) {
		compact := responseData[document](t, check("GET", "/forecasts/solar-wind-speed", "", 200))
		if compact["meta"] == nil || compact["horizon_hours"] != float64(1) {
			t.Fatalf("invalid forecast: %v", compact)
		}
		points := compact["predictions"].([]any)
		speed := points[0].(map[string]any)["variables"].(map[string]any)["v"].(map[string]any)
		if speed["continuous"].(map[string]any)["q90"] != float64(462) || len(speed["binary"].([]any)) != 3 {
			t.Fatalf("invalid forecast point: %v", speed)
		}
		geomagnetic := responseData[publicforecast.Forecast](t, check("GET", "/forecasts/geomagnetic-activity", "", 200))
		point := geomagnetic.Predictions[0]
		if point.UpdateIssuedAt == nil || !point.UpdateIssuedAt.Equal(point.Valid.Add(-49*time.Minute-3*time.Second)) ||
			point.Variables["ap"].Continuous.Q50 != 48 {
			t.Fatalf("fast L1 update was not forwarded: %+v", point)
		}
		check("GET", "/forecasts/geomagnetic-fast-l1", "", 404)
		hmf := responseData[publicforecast.Forecast](t, check("GET", "/forecasts/hmf", "", 200))
		point = hmf.Predictions[0]
		bt, bs := point.Variables["bt"], point.Variables["bs"]
		if point.UpdateIssuedAt != nil || bt.UpdateIssuedAt == nil ||
			!bt.UpdateIssuedAt.Equal(point.Valid.Add(-49*time.Minute-3*time.Second)) ||
			bt.Binary[0].Probability != .8 || bs.UpdateIssuedAt != nil || bs.Binary[0].Probability != .1 {
			t.Fatalf("fast L1 Bt update was not forwarded: %+v", point)
		}
		check("GET", "/forecasts/hmf-fast-l1", "", 404)
		metrics := responseData[document](t, check("GET", "/forecasts/solar-wind-speed/metrics", "", 200))
		if metrics["meta"] == nil {
			t.Fatal("metrics metadata was not forwarded")
		}
		check("GET", "/forecasts/hmf/metrics", "", 503)
		check("GET", "/forecasts/dst/verification", "", 200)
		check("GET", "/forecasts/solar-wind-events", "", 200)
		density := responseData[document](t, check("GET", "/forecasts/atmospheric-density", "", 200))
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
		if compact["meta"] == nil {
			t.Fatal("missing drag assessment details")
		}
		check("POST", "/risks/leo-drag", input+`,"horizon_hours":null}`, 422)
		check("POST", "/risks/leo-drag", input+`,"thresholds":{"elevated_altitude_loss_m":0,"high_altitude_loss_m":3}}`, 422)
		check("POST", "/risks/leo-drag", input+`,"thresholds":{"high_altitude_loss_m":3}}`, 422)
		got := responseData[document](t, check("POST", "/risks/leo-drag", input+`,"thresholds":null}`, 200))
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

func dragAssessmentFixture(input LeoDragRequest, issue time.Time) document {
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
	data["meta"] = document{"model": "circular_leo_drag_v1", "source": document{"model": "JB2008"}}
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
