package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

func regularHMFRelease(id string, issue time.Time, targets ...time.Time) release {
	bt, bs := []string{}, []string{}
	for _, target := range targets {
		clock := fmt.Sprintf("%s,%s,%d", issue.Format(time.RFC3339), target.Format(time.RFC3339), (target.Sub(issue) / time.Hour))
		bt = append(bt, clock+",0.5,0.3,0.1")
		bs = append(bs, clock+",0.2,0.1,0.05")
	}
	return release{ID: id, Product: hmfProduct, IssueTime: issue, Artifacts: []artifact{
		fastTestArtifact("hmf_total_threshold", "issue_time,valid_time,lead_hours,p_bt_ge_5,p_bt_ge_10,p_bt_ge_15", bt...),
		fastTestArtifact("hmf_southward_threshold", "issue_time,valid_time,lead_hours,p_bs_ge_5,p_bs_ge_10,p_bs_ge_15", bs...),
	}}
}

func fastBtTestRelease(issue, published time.Time, targets ...time.Time) release {
	rows := []string{}
	for _, target := range targets {
		rows = append(rows, issue.Format(time.RFC3339)+","+target.Format(time.RFC3339)+",0.9,0.8,0.6")
	}
	return release{ID: "fast-bt", Product: hmfFastProduct, IssueTime: issue, PublishedAt: published,
		Artifacts: []artifact{fastTestArtifact(hmfFastArtifact,
			"issue_time,valid_time,p_bt_ge_5,p_bt_ge_10,p_bt_ge_15", rows...)}}
}

func hmfProjection(t *testing.T, cache *preparedForecasts, now time.Time) publicforecast.Forecast {
	t.Helper()
	result, synced := cache.getAt(hmfProduct, now)
	var forecast publicforecast.Forecast
	if !synced || json.Unmarshal(result.body, &forecast) != nil {
		t.Fatalf("unavailable regular forecast: %s", result.body)
	}
	return forecast
}

func assertBtPoint(t *testing.T, point *publicforecast.ForecastPoint, issue *time.Time) {
	t.Helper()
	bt, bs := point.Variables["bt"], point.Variables["bs"]
	want := .3
	if issue != nil {
		want = .8
	}
	if bt.Binary[1].Probability != want || bs.Binary[1].Probability != .1 || point.UpdateIssuedAt != nil || bs.UpdateIssuedAt != nil {
		t.Fatalf("wrong variable-specific update: %+v, Bt %+v, Bs %+v", point, bt, bs)
	}
	if issue == nil && bt.UpdateIssuedAt != nil ||
		issue != nil && (bt.UpdateIssuedAt == nil || !bt.UpdateIssuedAt.Equal(*issue)) {
		t.Fatalf("unexpected Bt update time %v, expected %v", bt.UpdateIssuedAt, issue)
	}
}

func TestFastBtPreparedProjectionExpiresEachHour(t *testing.T) {
	baseIssue := time.Date(2026, time.November, 26, 0, 0, 0, 0, time.UTC)
	issue := baseIssue.Add(75 * time.Minute)
	published := issue.Add(30 * time.Second)
	targets := []time.Time{baseIssue.Add(2 * time.Hour), baseIssue.Add(3 * time.Hour), baseIssue.Add(6 * time.Hour)}
	source := &fastProjectionSource{reads: map[string]int{}, releases: map[string]release{
		hmfProduct:     regularHMFRelease("hmf-base", baseIssue, targets...),
		hmfFastProduct: fastBtTestRelease(issue, published, targets...),
	}}
	cache := newPreparedForecasts(source)
	cache.refresh(context.Background())
	cache.refresh(context.Background())
	for _, scenario := range []struct {
		at           time.Time
		firstUpdated int
	}{
		{published.Add(-time.Nanosecond), len(targets)},
		{published, 0},
		{targets[0], 1},
		{targets[1], 2},
		{targets[2], len(targets)},
	} {
		forecast := hmfProjection(t, cache, scenario.at)
		if !forecast.IssueTime.Equal(baseIssue) || forecast.HorizonHours != 6 || len(forecast.Predictions) != len(targets) {
			t.Fatalf("changed hourly forecast clock: %+v", forecast)
		}
		for i, point := range forecast.Predictions {
			if !point.Valid.Equal(targets[i]) || point.Lead != int(targets[i].Sub(baseIssue)/time.Hour) {
				t.Fatalf("changed hourly forecast target %d: %+v", i, point)
			}
			var expected *time.Time
			if i >= scenario.firstUpdated {
				expected = &issue
			}
			assertBtPoint(t, point, expected)
		}
	}
	if source.reads[hmfProduct] != 1 || source.reads[hmfFastProduct] != 1 {
		t.Fatalf("cached releases were reloaded: %v", source.reads)
	}
	nextIssue := baseIssue.Add(time.Hour)
	source.releases[hmfProduct] = regularHMFRelease("hmf-next", nextIssue, targets...)
	cache.refresh(context.Background())
	forecast := hmfProjection(t, cache, published)
	if !forecast.IssueTime.Equal(nextIssue) || forecast.Predictions[0].Lead != 1 || source.reads[hmfFastProduct] != 1 {
		t.Fatalf("hourly rollover failed or fast release reloaded: %+v", forecast)
	}
	assertBtPoint(t, forecast.Predictions[0], &issue)

	invalid := fastBtTestRelease(issue, published, targets...)
	invalid.ID = "bad"
	invalid.Artifacts[0].CSVText = strings.ReplaceAll(invalid.Artifacts[0].CSVText, ",0.9,0.8,", ",0.1,0.8,")
	source.releases[hmfFastProduct] = invalid
	cache.refresh(context.Background())
	cache.refresh(context.Background())
	assertBtPoint(t, hmfProjection(t, cache, published).Predictions[0], nil)
	if source.reads[hmfFastProduct] != 2 {
		t.Fatalf("invalid release was prepared again: %v", source.reads)
	}
}

func TestFastBtCurrentHourCannotBeUpdated(t *testing.T) {
	issue := time.Date(2026, time.November, 26, 1, 0, 0, 0, time.UTC)
	start := issue.Add(-time.Hour)
	next := issue.Add(time.Hour)
	fast := fastBtTestRelease(issue, issue, issue, next)
	source := &fastProjectionSource{reads: map[string]int{}, releases: map[string]release{
		hmfProduct:     regularHMFRelease("hmf-base", start, issue, next),
		hmfFastProduct: fast,
	}}
	cache := newPreparedForecasts(source)
	cache.refresh(context.Background())
	forecast := hmfProjection(t, cache, issue)
	assertBtPoint(t, forecast.Predictions[0], nil)
	assertBtPoint(t, forecast.Predictions[1], &issue)
	for _, bad := range []release{
		fastBtTestRelease(issue, issue.Add(-time.Second), next),
		fastBtTestRelease(issue, issue, next.Add(time.Minute)),
		fastBtTestRelease(issue, issue, next.Add(-2*time.Hour)),
		fastBtTestRelease(issue, issue, issue.Add(6*time.Hour)),
		fastBtTestRelease(issue, issue, next, next),
	} {
		if _, err := hmfFastSpec.prepare(bad); err == nil {
			t.Fatalf("invalid input accepted: %+v", bad)
		}
	}
}

func TestFastBtIsOnlyAvailableThroughRegularPublicForecast(t *testing.T) {
	handler := newHandler(emptyStore{}, "token", "", 6, newPreparedForecasts(nil), metricReports{})
	for _, path := range []string{preparedForecastPath + hmfFastProduct, "/internal/v1/forecasts/" + hmfFastProduct + "/verification"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		req.Header.Set("Authorization", "Bearer token")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		if response.Code != http.StatusNotFound {
			t.Fatalf("%s returned %d: %s", path, response.Code, response.Body)
		}
	}
}
