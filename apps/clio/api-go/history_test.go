package main

import (
	"testing"
	"time"

	"github.com/a-kryvenko/argus/observationcatalog"
)

func TestNativeHourlyMeansPreserveGapsAndTrimWindowEdges(t *testing.T) {
	firstHour := time.Date(2026, 11, 24, 9, 0, 0, 0, time.UTC)
	start := firstHour.Add(-30 * time.Minute)
	end := firstHour.Add(2*time.Hour + 30*time.Minute)
	low, high := 400.0, 500.0
	receipt := firstHour.Add(55 * time.Minute)
	points := aggregateWind([]observationcatalog.Point{
		{ObservedAt: start, Value: &low}, // Starts before the selected range.
		{ObservedAt: firstHour.Add(12 * time.Minute), Value: &low, ReceivedAt: &receipt},
		{ObservedAt: firstHour.Add(21 * time.Minute), Quality: "flagged"},
		{ObservedAt: firstHour.Add(56 * time.Minute), Value: &high},
		{ObservedAt: end.Add(-5 * time.Minute), Value: &high}, // Ends after the selected range.
	}, start, end, time.Hour)
	if len(points) != 1 {
		t.Fatalf("expected only the observed, complete UTC hour, got %v", points)
	}
	point := points[0]
	if point.Value == nil || *point.Value != 450 || *point.Count != 2 || *point.ExpectedCount != 60 {
		t.Fatalf("unexpected mean or coverage: %+v", point)
	}
	if point.ReceivedAt != nil {
		t.Fatalf("some contributing minutes have unknown receipt time: %+v", point)
	}
}

func TestAggregateAvailabilityIncludesRetractions(t *testing.T) {
	hour := time.Date(2026, 11, 24, 9, 0, 0, 0, time.UTC)
	received, retracted := hour.Add(time.Minute), hour.Add(2*time.Hour)
	value := 400.0
	points := aggregateWind([]observationcatalog.Point{
		{ObservedAt: hour, Value: &value, ReceivedAt: &received},
		{ObservedAt: hour.Add(time.Minute), Quality: "flagged", ReceivedAt: &retracted},
	}, hour, hour.Add(time.Hour), time.Hour)
	if len(points) != 1 || *points[0].Count != 1 || points[0].ReceivedAt == nil || !points[0].ReceivedAt.Equal(retracted) {
		t.Fatalf("aggregate must reflect receipt of the retraction: %+v", points)
	}
}

func TestArchivedTotalFieldRequiresAllComponents(t *testing.T) {
	x, y, z := 3.0, -4.0, 12.0
	row := map[string]*float64{"bx": &x, "by": &y}
	if value := totalField(row); value != nil {
		t.Fatalf("total field without Bz = %v", *value)
	}
	row["bz"] = &z
	if value := totalField(row); value == nil || *value != 13 {
		t.Fatalf("total field = %v, expected magnitude of archived hourly vector to be 13", value)
	}
}
