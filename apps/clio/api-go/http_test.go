package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
	"time"
)

const latestPath = "/internal/v1/observations/latest"
const historyPath = "/internal/v1/observations/history"

type stubReader struct {
	observationReader
	rows          []measurement
	selected      []string
	interval      string
	timeBasis     string
	end           time.Time
	historyPoints numericPoints
	err           error
}

func (db *stubReader) ready(context.Context) error {
	return db.err
}

func (db *stubReader) latest(_ context.Context, metrics []string) ([]measurement, error) {
	db.selected = metrics
	return db.rows, db.err
}

func (db *stubReader) history(_ context.Context, metrics []string, _, end time.Time, interval, timeBasis string) (numericPoints, error) {
	db.selected, db.interval, db.timeBasis, db.end = metrics, interval, timeBasis, end
	return db.historyPoints, db.err
}

func request(handler http.Handler, path, authorization string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.Header.Set("Authorization", authorization)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, req)
	return response
}

func TestLatest(t *testing.T) {
	at := time.Date(2026, 11, 24, 12, 0, 0, 0, time.UTC)
	bz, dst, missing := -2.5, -10.75, math.Inf(1)
	db := &stubReader{rows: []measurement{
		{metric: "n", observedAt: at.Add(time.Minute), value: &missing},
		{metric: "bz", observedAt: at, value: &bz},
		{metric: "dst", observedAt: at, value: &dst},
	}}
	response := request(newHandler(db, "secret", ""), latestPath+"?metrics=n,bz,%20dst,bz", "bEaReR secret")
	if response.Code != http.StatusOK || response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("unexpected response: %d %s", response.Code, response.Body)
	}
	if !slices.Equal(db.selected, []string{"n", "bz", "dst"}) {
		t.Fatalf("selected metrics = %v", db.selected)
	}
	want := `{"success":true,"data":{"points":[{"bz":-2.5,"dst":-10.75,"time":"2026-11-24T12:00:00Z"},{"n":null,"time":"2026-11-24T12:01:00Z"}]},"error":null}` + "\n"
	if response.Body.String() != want {
		t.Fatalf("response = %s, want %s", response.Body, want)
	}
}

func TestHistoryTimeBasisAndCadences(t *testing.T) {
	const dates = "?from=2026-11-24T00:00:00Z&to=2026-11-24T01:00:00Z"
	for _, tc := range []struct {
		name, query, interval, timeBasis string
		selected                         []string
	}{
		{"legacy hourly", "", "1h", "", legacyMetrics},
		{"native hourly", "&metrics=bz,v&interval=1h&time_basis=l1", "1h", l1TimeBasis, []string{"bz", "v"}},
		{"native 5m", "&interval=5m&time_basis=l1", "5m", l1TimeBasis, windMetrics},
		{"near Earth", "&metrics=bt&time_basis=near_earth", "1h", nearEarthTimeBasis, []string{"bt"}},
		{"X-ray", "&metrics=xrs_a,xrs_b&interval=1m", "1m", "", xrsMetrics},
		{"protons", "&metrics=proton_10mev,proton_100mev&interval=5m", "5m", "", protonMetrics},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db := &stubReader{}
			response := request(newHandler(db, "secret", ""), historyPath+dates+tc.query, "Bearer secret")
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", response.Code, response.Body)
			}
			if db.interval != tc.interval || db.timeBasis != tc.timeBasis || !slices.Equal(db.selected, tc.selected) {
				t.Fatalf("interval=%s, time basis=%s, metrics=%v", db.interval, db.timeBasis, db.selected)
			}
			var body struct {
				Data map[string]any `json:"data"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if basis, ok := body.Data["time_basis"]; ok != (tc.timeBasis != "") || ok && basis != tc.timeBasis {
				t.Fatalf("unexpected time basis in response: %s", response.Body)
			}
		})
	}
	for _, query := range []string{
		"&metrics=v&time_basis=", "&metrics=v&time_basis=earth", "&metrics=v&interval=1m&time_basis=near_earth",
		"&metrics=v,dst&time_basis=l1", "&metrics=xrs_b&interval=5m", "&metrics=proton_10mev&interval=1h",
	} {
		t.Run(query, func(t *testing.T) {
			db := &stubReader{}
			response := request(newHandler(db, "secret", ""), historyPath+dates+query, "Bearer secret")
			if response.Code != http.StatusUnprocessableEntity || db.selected != nil {
				t.Fatalf("invalid request reached history reader: %d %s", response.Code, response.Body)
			}
		})
	}
}

func TestGOESHistoryPreservesFractionalFlux(t *testing.T) {
	at := time.Date(2026, 11, 24, 12, 10, 0, 0, time.UTC)
	protons := 0.31
	db := &stubReader{historyPoints: numericPoints{at: {values: map[string]*float64{"proton_10mev": &protons}}}}
	query := "?from=2026-11-24T12:00:00Z&to=2026-11-24T13:00:00Z&metrics=proton_10mev&interval=5m"
	response := request(newHandler(db, "secret", ""), historyPath+query, "Bearer secret")
	want := `{"success":true,"data":{"interval":"5m","points":[{"proton_10mev":0.31,"time":"2026-11-24T12:10:00Z"}]},"error":null}` + "\n"
	if response.Code != http.StatusOK || response.Body.String() != want {
		t.Fatalf("unexpected response: %d %s", response.Code, response.Body)
	}
}

func TestL1AggregatesStopAtCurrentTime(t *testing.T) {
	for _, interval := range []string{"5m", "1h"} {
		t.Run(interval, func(t *testing.T) {
			start := time.Now().UTC().Add(-time.Hour)
			end := start.Add(2 * time.Hour)
			db := &stubReader{}
			query := "?metrics=v&time_basis=l1&interval=" + interval + "&from=" + start.Format(time.RFC3339) + "&to=" + end.Format(time.RFC3339)
			before := time.Now().UTC()
			response := request(newHandler(db, "secret", ""), historyPath+query, "Bearer secret")
			if response.Code != http.StatusOK || db.end.Before(before) || db.end.After(time.Now().UTC()) {
				t.Fatalf("L1 %s query ended at %s: %d %s", interval, db.end, response.Code, response.Body)
			}
		})
	}
}

func TestStatusCodes(t *testing.T) {
	for _, tc := range []struct {
		name, path, token, auth, errorCode string
		dbError                            error
		status                             int
	}{
		{name: "live", path: "/health/live", status: 200, dbError: errors.New("database offline")},
		{name: "ready", path: "/health/ready", status: 200},
		{name: "not ready", path: "/health/ready", status: 503, errorCode: "NOT_READY", dbError: errors.New("database offline")},
		{name: "missing configured token", path: latestPath, status: 503, errorCode: "NOT_READY"},
		{name: "wrong token", path: latestPath, token: "secret", auth: "Bearer wrong", status: 401, errorCode: "HTTP_ERROR"},
		{name: "empty metrics", path: latestPath + "?metrics=", token: "secret", auth: "Bearer secret", status: 422, errorCode: "HTTP_ERROR"},
		{name: "malformed metrics", path: latestPath + "?metrics=bz;dst", token: "secret", auth: "Bearer secret", status: 422, errorCode: "HTTP_ERROR"},
		{name: "storage failure", path: latestPath, token: "secret", auth: "Bearer secret", status: 503, errorCode: "NOT_READY", dbError: errors.New("database offline")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := request(newHandler(&stubReader{err: tc.dbError}, tc.token, ""), tc.path, tc.auth)
			if response.Code != tc.status {
				t.Fatalf("status = %d, want %d", response.Code, tc.status)
			}
			if tc.errorCode == "" {
				return
			}
			var body apiResponse
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if body.Success || body.Error == nil || body.Error.Code != tc.errorCode || body.Error.Message == "database offline" {
				t.Fatalf("unexpected error response: %s", response.Body)
			}
		})
	}
}
