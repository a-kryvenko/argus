package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

type fastProjectionSource struct {
	releases map[string]release
	reads    map[string]int
	readErr  error
}

func (s *fastProjectionSource) latestReleaseIDs(context.Context) (map[string]string, error) {
	ids := make(map[string]string, len(s.releases))
	for name, source := range s.releases {
		ids[name] = source.ID
	}
	return ids, nil
}

func (s *fastProjectionSource) readRelease(_ context.Context, product, id string) (release, error) {
	s.reads[product]++
	if product == geomagneticFastProduct && s.readErr != nil {
		return release{}, s.readErr
	}
	source := s.releases[product]
	if source.ID != id {
		return release{}, errors.New("unexpected release ID")
	}
	return source, nil
}

func fastTestArtifact(name, header string, rows ...string) artifact {
	return artifact{Name: name, Columns: strings.Split(header, ","), RowCount: len(rows),
		CSVText: header + "\n" + strings.Join(rows, "\n") + "\n"}
}

func regularGeomagneticRelease(id string, issue time.Time, targets ...time.Time) release {
	kp, ap := []string{}, []string{}
	for _, target := range targets {
		clock := fmt.Sprintf("%s,%s,%d", issue.Format(time.RFC3339), target.Format(time.RFC3339), (target.Sub(issue) / time.Hour))
		kp = append(kp, clock+",0.5,0.3,0.1")
		ap = append(ap, clock+",8,12,20")
	}
	return release{ID: id, Product: geomagneticProduct, IssueTime: issue, Artifacts: []artifact{
		fastTestArtifact("kp_threshold", "issue_time,valid_time,lead_hours,p_kp_ge_4,p_kp_ge_5,p_kp_ge_6", kp...),
		fastTestArtifact("ap_quantile", "issue_time,valid_time,lead_hours,ap_q10,ap_q50,ap_q90", ap...),
	}}
}

func fastGeomagneticTestRelease(issue, published time.Time, targets ...time.Time) release {
	rows := []string{}
	for _, target := range targets {
		rows = append(rows, issue.Format(time.RFC3339)+","+target.Format(time.RFC3339)+",0.9,0.8,0.6,12,32,80")
	}
	return release{ID: "fast", Product: geomagneticFastProduct, IssueTime: issue, PublishedAt: published,
		Artifacts: []artifact{fastTestArtifact(geomagneticFastArtifact,
			"issue_time,valid_time,p_kp_ge_4,p_kp_ge_5,p_kp_ge_6,ap_q10,ap_q50,ap_q90", rows...)}}
}

func projectedGeomagnetic(t *testing.T, cache *preparedForecasts, now time.Time) publicforecast.Forecast {
	t.Helper()
	result, synced := cache.getAt(geomagneticProduct, now)
	var forecast publicforecast.Forecast
	if !synced || json.Unmarshal(result.body, &forecast) != nil {
		t.Fatalf("unavailable regular forecast: %s", result.body)
	}
	return forecast
}

func assertGeomagneticPoint(t *testing.T, point *publicforecast.ForecastPoint, expectedIssue *time.Time) {
	t.Helper()
	p5, q50 := .3, 12.
	if expectedIssue != nil {
		p5, q50 = .8, 32.
	}
	if point.Variables["kp"].Binary[1].Probability != p5 || point.Variables["ap"].Continuous.Q50 != q50 {
		t.Fatalf("inconsistent Kp/ap overlay: %+v", point)
	}
	if expectedIssue == nil && point.UpdateIssuedAt != nil ||
		expectedIssue != nil && (point.UpdateIssuedAt == nil || !point.UpdateIssuedAt.Equal(*expectedIssue)) {
		t.Fatalf("unexpected update issue time %v, expected %v", point.UpdateIssuedAt, expectedIssue)
	}
}

func TestFastGeomagneticPreparedProjection(t *testing.T) {
	baseIssue := time.Date(2026, time.November, 26, 0, 0, 0, 0, time.UTC)
	target := baseIssue.Add(3 * time.Hour)
	issue := baseIssue.Add(62 * time.Minute)
	published := issue.Add(30 * time.Second)
	next := target.Add(time.Hour)
	source := &fastProjectionSource{reads: map[string]int{}, releases: map[string]release{
		geomagneticProduct:     regularGeomagneticRelease("base", baseIssue, target, next),
		geomagneticFastProduct: fastGeomagneticTestRelease(issue, published, target),
	}}
	cache := newPreparedForecasts(source)
	cache.refresh(context.Background())
	cache.refresh(context.Background())
	for _, tc := range []struct {
		at     time.Time
		update *time.Time
	}{
		{published.Add(-time.Nanosecond), nil},
		{published, &issue},
		{target.Add(-time.Nanosecond), &issue},
		{target, nil},
	} {
		forecast := projectedGeomagnetic(t, cache, tc.at)
		if !forecast.IssueTime.Equal(baseIssue) || len(forecast.Predictions) != 2 ||
			!forecast.Predictions[0].Valid.Equal(target) || forecast.Predictions[0].Lead != 3 {
			t.Fatalf("changed hourly forecast clock: %+v", forecast)
		}
		assertGeomagneticPoint(t, forecast.Predictions[0], tc.update)
		assertGeomagneticPoint(t, forecast.Predictions[1], nil)
	}
	if source.reads[geomagneticFastProduct] != 1 || source.reads[geomagneticProduct] != 1 {
		t.Fatalf("cached releases were reloaded: %v", source.reads)
	}

	// Reproject on a new regular release without re-reading the same fast release.
	nextIssue := baseIssue.Add(time.Hour)
	source.releases[geomagneticProduct] = regularGeomagneticRelease("next", nextIssue, target, next)
	cache.refresh(context.Background())
	forecast := projectedGeomagnetic(t, cache, published)
	if !forecast.IssueTime.Equal(nextIssue) || forecast.Predictions[0].Lead != 2 ||
		source.reads[geomagneticFastProduct] != 1 {
		t.Fatalf("hourly rollover failed or fast release reloaded: %+v", forecast)
	}
	assertGeomagneticPoint(t, forecast.Predictions[0], &issue)

	// A corrupt replacement cannot hide the regular forecast or retain old fast values.
	invalid := fastGeomagneticTestRelease(issue, published, target)
	invalid.ID = "bad"
	invalid.Artifacts[0].CSVText = strings.ReplaceAll(invalid.Artifacts[0].CSVText, ",0.9,0.8,", ",0.1,0.8,")
	source.releases[geomagneticFastProduct] = invalid
	cache.refresh(context.Background())
	cache.refresh(context.Background())
	regular := projectedGeomagnetic(t, cache, published)
	assertGeomagneticPoint(t, regular.Predictions[0], nil)
	if source.reads[geomagneticFastProduct] != 2 {
		t.Fatalf("invalid release was prepared again: %v", source.reads)
	}
}

func TestFastGeomagneticLatePublicationAndRetry(t *testing.T) {
	baseIssue := time.Date(2026, time.November, 26, 0, 0, 0, 0, time.UTC)
	boundary := baseIssue.Add(3 * time.Hour)
	next := boundary.Add(3 * time.Hour)
	published := boundary.Add(time.Second)
	source := &fastProjectionSource{
		reads: map[string]int{}, readErr: errors.New("temporary storage error"),
		releases: map[string]release{
			geomagneticProduct:     regularGeomagneticRelease("base", baseIssue, boundary, next),
			geomagneticFastProduct: fastGeomagneticTestRelease(boundary, published, boundary, next),
		},
	}
	cache := newPreparedForecasts(source)
	cache.refresh(context.Background())
	forecast := projectedGeomagnetic(t, cache, published)
	assertGeomagneticPoint(t, forecast.Predictions[0], nil)
	assertGeomagneticPoint(t, forecast.Predictions[1], nil)
	source.readErr = nil
	cache.refresh(context.Background())
	forecast = projectedGeomagnetic(t, cache, published)
	assertGeomagneticPoint(t, forecast.Predictions[0], nil)
	assertGeomagneticPoint(t, forecast.Predictions[1], &boundary)
	delete(source.releases, geomagneticFastProduct)
	cache.refresh(context.Background())
	forecast = projectedGeomagnetic(t, cache, published)
	assertGeomagneticPoint(t, forecast.Predictions[1], nil)
}

func TestFastGeomagneticMalformedRelease(t *testing.T) {
	issue := time.Date(2026, time.November, 26, 1, 15, 0, 0, time.UTC)
	published := issue.Add(time.Second)
	target := issue.Truncate(3 * time.Hour).Add(3 * time.Hour)
	for _, item := range []release{
		fastGeomagneticTestRelease(issue, issue.Add(-time.Second), target),
		fastGeomagneticTestRelease(issue, published, target.Add(-3*time.Hour)),
		fastGeomagneticTestRelease(issue, published, target.Add(time.Hour)),
		fastGeomagneticTestRelease(issue, published, target.Add(3*time.Hour)),
		fastGeomagneticTestRelease(issue, published, target, target),
	} {
		if _, err := prepareFastGeomagnetic(item); err == nil {
			t.Fatalf("invalid input accepted: %+v", item)
		}
	}
}

func TestFastGeomagneticOnlyThroughRegularPreparedProduct(t *testing.T) {
	target := time.Now().UTC().Truncate(3 * time.Hour).Add(3 * time.Hour)
	issue := target.Add(-3 * time.Hour).Add(time.Second)
	source := &fastProjectionSource{reads: map[string]int{}, releases: map[string]release{
		geomagneticProduct:     regularGeomagneticRelease("base", issue.Truncate(time.Hour), target),
		geomagneticFastProduct: fastGeomagneticTestRelease(issue, issue, target),
	}}
	cache := newPreparedForecasts(source)
	cache.refresh(context.Background())
	for targetName, status := range map[string]int{geomagneticProduct: http.StatusOK, geomagneticFastProduct: http.StatusNotFound} {
		request := httptest.NewRequest(http.MethodGet, preparedForecastPath+targetName, nil)
		request.Header.Set("Authorization", "Bearer token")
		response := httptest.NewRecorder()
		newHandler(emptyStore{}, "token", "", 6, cache, metricReports{}).ServeHTTP(response, request)
		if response.Code != status {
			t.Fatalf("%s returned %d: %s", targetName, response.Code, response.Body)
		}
	}
}
