package main

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

type smokeTransport func(*http.Request) (*http.Response, error)

func (transport smokeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return transport(req)
}

func TestSmokeChecksReadySlotAndLatestContract(t *testing.T) {
	for _, tc := range []struct {
		name, latest, wantError string
		status                  int
	}{
		{name: "empty points are valid", latest: `{"success":true,"data":{"points":[]}}`, status: 200},
		{name: "invalid points", latest: `{"success":true,"data":{"points":null}}`, status: 200, wantError: "invalid latest"},
		{name: "failed request", latest: `{}`, status: 503, wantError: "returned 503"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			readyCalls := 0
			client := &http.Client{Transport: smokeTransport(func(req *http.Request) (*http.Response, error) {
				headers := make(http.Header)
				status, body := http.StatusOK, `{}`
				switch req.URL.Path {
				case "/health/ready":
					if auth := req.Header.Get("Authorization"); auth != "" {
						t.Fatalf("unexpected readiness authorization: %s", auth)
					}
					readyCalls++
					slot := "clio-blue"
					if readyCalls == 2 {
						slot = "clio-green"
					}
					headers.Set("X-Argus-Upstream", slot)
				case "/internal/v1/observations/latest":
					if readyCalls != 2 || req.Header.Get("Authorization") != "Bearer secret" {
						t.Fatal("latest must be checked against the selected slot with the service token")
					}
					status, body = tc.status, tc.latest
				default:
					t.Fatalf("unexpected request: %s", req.URL)
				}
				return &http.Response{StatusCode: status, Header: headers, Body: io.NopCloser(strings.NewReader(body))}, nil
			})}
			err := smoke(client, "http://observations-router:8101", "clio-green", "secret")
			if tc.wantError == "" && err != nil {
				t.Fatal(err)
			}
			if tc.wantError != "" && (err == nil || !strings.Contains(err.Error(), tc.wantError)) {
				t.Fatalf("error = %v, want %q", err, tc.wantError)
			}
		})
	}
}
