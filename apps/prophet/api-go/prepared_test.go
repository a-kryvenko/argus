package main

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type projectionTestSource struct {
	latest release
	reads  int
}

func (s *projectionTestSource) latestReleaseIDs(context.Context) (map[string]string, error) {
	return map[string]string{s.latest.Product: s.latest.ID}, nil
}

func (s *projectionTestSource) readRelease(context.Context, string, string) (release, error) {
	s.reads++
	return s.latest, nil
}

func TestPreparedForecastsReuseImmutableRelease(t *testing.T) {
	const header = "issue_time,valid_time,lead_hours,dst_q10,dst_q50,dst_q90"
	const data = "2026-11-24T00:00:00Z,2026-11-24T01:00:00Z,1,-20,-10,0"
	source := &projectionTestSource{latest: release{
		ID: "first", Product: "dst", IssueTime: time.Date(2026, 11, 24, 0, 0, 0, 0, time.UTC),
		Artifacts: []artifact{{Name: "dst_quantile", Columns: strings.Split(header, ","), RowCount: 1,
			CSVText: header + "\n" + data + "\n"}},
	}}
	cache := newPreparedForecasts(source)
	cache.refresh(context.Background())
	cache.refresh(context.Background())
	handler := newHandler(emptyStore{}, "token", "", 6, cache, metricReports{})
	request := httptest.NewRequest("GET", preparedForecastPath+"dst", nil)
	request.Header.Set("Authorization", "Bearer token")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != 200 || !strings.Contains(response.Body.String(), `"q50":-10`) ||
		!strings.Contains(response.Body.String(), `"meta"`) {
		t.Fatalf("unexpected forecast: %d %s", response.Code, response.Body)
	}
	if source.reads != 1 {
		t.Fatalf("read %d releases, want one preparation", source.reads)
	}
	source.latest.ID = "invalid"
	source.latest.Artifacts[0].CSVText = "corrupt"
	cache.refresh(context.Background())
	cache.refresh(context.Background())
	record, synced := cache.get("dst")
	if !synced || record.body != nil || source.reads != 2 {
		t.Fatalf("invalid replacement fell back or was prepared twice: read %d, cache %v", source.reads, record)
	}
}

func TestPreparedDensityExpiresAtReadTime(t *testing.T) {
	cache := newPreparedForecasts(nil)
	cache.synced = true
	cache.latest[densityProduct] = preparedForecast{
		issueTime: time.Now().Add(-7 * time.Hour), body: []byte(`{"target":"atmospheric-density"}`),
	}
	handler := newHandler(emptyStore{}, "token", "", 6, cache, metricReports{})
	request := httptest.NewRequest("GET", preparedForecastPath+densityProduct, nil)
	request.Header.Set("Authorization", "Bearer token")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != 503 {
		t.Fatalf("expired density returned %d: %s", response.Code, response.Body)
	}
}
