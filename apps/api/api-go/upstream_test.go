package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/a-kryvenko/argus/observationcatalog"
)

type testTransport func(*http.Request) *http.Response

func (transport testTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return transport(r), nil
}

func upstreamReply(body string) *http.Response {
	return &http.Response{
		StatusCode:    http.StatusOK,
		Body:          io.NopCloser(strings.NewReader(body)),
		ContentLength: -1,
	}
}

func TestPublicationRejectsInvalidUpstreamData(t *testing.T) {
	const orbit = `{"altitude_km":400,"inclination_deg":51,"mass_kg":100,"effective_area_m2":1,"drag_coefficient":2}`
	for _, tc := range []struct{ method, path, request, reply string }{
		{"GET", "/summary", "", "{}"},
		{"GET", "/summary", "", `{"generated_at":"2026-01-01T00:00:00Z","observations":null}`},
		{"GET", "/forecasts/solar-wind-speed", "", "null"},
		{"GET", "/forecasts/solar-wind-speed", "", `{"target":"dst","issue_time":"2026-01-01T00:00:00Z","available_variables":["v"],"horizon_hours":1,"predictions":[{"lead_hours":1,"valid_time":"2026-01-01T01:00:00Z","variables":{"v":{}}}]}`},
		{"GET", "/forecasts/solar-wind-speed", "", `{"target":"solar-wind-speed","issue_time":"2026-01-01T00:00:00Z","available_variables":["v"],"horizon_hours":1,"predictions":[{"lead_hours":1,"valid_time":"2026-01-01T02:00:00Z","variables":{"v":{}}}]}`},
		{"GET", "/forecasts/solar-wind-speed/metrics", "", `{"target":"dst","variables":{}}`},
		{"GET", "/forecasts/solar-wind-events", "", `{"product":"solar-wind-events","issue_time":"2026-01-01T00:00:00Z","schema_version":1}`},
		{"GET", "/forecasts/atmospheric-density", "", `{"target":"atmospheric-density","predictions":[{}]}`},
		{"POST", "/risks/leo-drag", orbit, `{"inputs": {"altitude_km":500},"predictions":[]}`},
	} {
		t.Run(tc.path+"/"+tc.reply, func(t *testing.T) {
			owner := upstream{"http://owner", "token"}
			app := &api{prophet: owner, intelligence: owner,
				client: &http.Client{Transport: testTransport(func(_ *http.Request) *http.Response {
					return upstreamReply(tc.reply)
				})}}
			w := httptest.NewRecorder()
			app.routes().ServeHTTP(w, httptest.NewRequest(tc.method, prefix+tc.path, strings.NewReader(tc.request)))
			if w.Code != http.StatusServiceUnavailable || w.Header().Get("Cache-Control") != "no-store" ||
				!strings.Contains(w.Body.String(), `"code":"NOT_READY"`) {
				t.Fatalf("malformed publication reached the client: %d %s", w.Code, w.Body)
			}
		})
	}
}

func TestRequestRejectsOverlongUpstreamBody(t *testing.T) {
	owner := upstream{"http://owner", "token"}
	app := &api{client: &http.Client{Transport: testTransport(func(_ *http.Request) *http.Response {
		return upstreamReply(`{"very":"long response"}`)
	})}}
	body, _, err := app.request(context.Background(), owner, "GET", "/report", nil, nil, time.Second, 5)
	if err == nil || body != nil {
		t.Fatalf("read beyond response limit: %v %q", err, body)
	}
}

func TestObservationHistoryClipsProducerResultsAndPreservesDailyCoverage(t *testing.T) {
	owner := upstream{"http://clio", "token"}
	from := time.Now().UTC().Truncate(24 * time.Hour).Add(-38 * time.Hour)
	to := from.Add(24 * time.Hour)
	payload, err := json.Marshal(document{"success": true, "data": map[string]observationcatalog.Series{
		"f10_7": {TimeBasis: "measurement", Aggregation: "source", ResolutionSeconds: 86400, Points: []observationcatalog.Point{
			{ObservedAt: from.Add(-time.Hour), SourceProduct: "celestrak.f107", Value: pointer(100.0)},
			{ObservedAt: from.Add(10 * time.Hour), SourceProduct: "celestrak.f107", Value: pointer(103.0)},
			{ObservedAt: to, SourceProduct: "celestrak.f107", Value: pointer(110.0)},
		}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var propagated bool
	app := &api{clio: owner, client: &http.Client{Transport: testTransport(func(r *http.Request) *http.Response {
		propagated = r.URL.Query().Has("time_basis")
		return upstreamReply(string(payload))
	})}}
	w := httptest.NewRecorder()
	app.routes().ServeHTTP(w, httptest.NewRequest("GET", prefix+"/observations/history"+
		"?observations=f10_7&time_basis=near_earth&resolution=1m&from="+from.Format(time.RFC3339)+"&to="+to.Format(time.RFC3339), nil))
	if w.Code != 200 || propagated {
		t.Fatalf("non-wind history forwarded a wind basis: %d %s", w.Code, w.Body)
	}
	data := responseData[struct {
		Series map[string]SolarHistorySeries `json:"series"`
	}](t, w)
	daily := data.Series["f10_7"]
	if len(daily.Points) != 1 || daily.Points[0].Value == nil || *daily.Points[0].Value != 103 ||
		daily.Coverage.Basis != dailyCoverage || daily.Coverage.ExpectedSlots != 1 ||
		daily.Coverage.UsableSlots != 1 || daily.Coverage.Percent == nil || *daily.Coverage.Percent != 100 {
		t.Fatalf("daily output outside the half-open history window or inconsistent coverage: %+v", daily)
	}
}

func TestObservationBoundaryRejectsWrongCadenceAndAggregateCount(t *testing.T) {
	owner := upstream{"http://clio", "token"}
	from := time.Date(2026, 11, 20, 10, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		name, basis, source string
		resolution, count   int
	}{
		{"wrong cadence", l1TimeBasis, "swpc.rtsw_mag", 5, 20},
		{"invalid count", l1TimeBasis, "swpc.rtsw_mag", 3600, 61},
		{"mixed time basis", l1TimeBasis, "omni.hourly", 3600, 20},
		{"wrong declared basis", nearEarthTimeBasis, "omni.hourly", 3600, 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload, err := json.Marshal(document{"success": true, "data": map[string]observationcatalog.Series{
				"bz": {TimeBasis: tc.basis, Aggregation: "mean", ResolutionSeconds: tc.resolution,
					Points: []observationcatalog.Point{{ObservedAt: from, SourceProduct: tc.source, Value: pointer(-5.0),
						Count: pointer(tc.count), ExpectedCount: pointer(60)}}},
			}})
			if err != nil {
				t.Fatal(err)
			}
			app := &api{clio: owner, client: &http.Client{Transport: testTransport(func(_ *http.Request) *http.Response {
				return upstreamReply(string(payload))
			})}}
			if _, err := app.numeric(context.Background(), []string{"bz"}, from, from.Add(time.Hour), "1h", l1TimeBasis); err == nil {
				t.Fatal("invalid aggregate accepted")
			}
		})
	}
}

func TestObservationChangesUseTheReturnedSampleAndExactIndexInterval(t *testing.T) {
	now := time.Date(2026, 11, 20, 10, 15, 0, 0, time.UTC)
	wind := sample{ObservedAt: now.Add(-time.Minute), Value: pointer(7.0)}
	past := []sample{
		{ObservedAt: wind.ObservedAt.Add(-time.Hour), Value: pointer(4.0)},
		{ObservedAt: wind.ObservedAt, Value: pointer(40.0)},
		{ObservedAt: now, Value: pointer(70.0)},
	}
	if got := windChange(wind, past, now, 10*time.Minute); got == nil || *got != 3 {
		t.Fatalf("trend used a revision or point outside the latest snapshot: %v", got)
	}
	index := sample{ObservedAt: now.Truncate(time.Hour), Value: pointer(-12.0)}
	past = []sample{{ObservedAt: index.ObservedAt.Add(-2 * time.Hour), Value: pointer(-6.0)}}
	if got := intervalChange(index, past, now, time.Hour, 2*time.Hour); got != nil {
		t.Fatalf("trend used an interval which had already ended an hour ago: %v", got)
	}
}
