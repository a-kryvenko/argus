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

type stubReader struct {
	observationReader
	rows     []measurement
	selected []string
	err      error
}

func (db *stubReader) ready(context.Context) error {
	return db.err
}

func (db *stubReader) latest(_ context.Context, metrics []string) ([]measurement, error) {
	db.selected = metrics
	return db.rows, db.err
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
	want := `{"success":true,"data":{"points":[{"bz":-2.5,"dst":-10,"time":"2026-11-24T12:00:00Z"},{"n":null,"time":"2026-11-24T12:01:00Z"}]},"error":null}` + "\n"
	if response.Body.String() != want {
		t.Fatalf("response = %s, want %s", response.Body, want)
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
