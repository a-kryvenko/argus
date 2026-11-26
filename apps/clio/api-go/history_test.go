package main

import (
	"testing"
	"time"
)

func TestNativeHourlyMeansPreserveGapsAndTrimWindowEdges(t *testing.T) {
	firstHour := time.Date(2026, 11, 24, 9, 0, 0, 0, time.UTC)
	start := firstHour.Add(-30 * time.Minute)
	end := firstHour.Add(2*time.Hour + 30*time.Minute)
	low, high := 400.0, 500.0
	buckets := make(historyBuckets)
	for _, row := range []measurement{
		{"v", start, &low}, // Starts before the selected range.
		{"v", firstHour.Add(12 * time.Minute), &low},
		{"v", firstHour.Add(21 * time.Minute), nil},
		{"v", firstHour.Add(56 * time.Minute), &high},
		{"v", end.Add(-5 * time.Minute), &high}, // Ends after the selected range.
	} {
		buckets.add(row, start, end, time.Hour)
	}
	if len(buckets) != 1 {
		t.Fatalf("expected only the observed, complete UTC hour, got %v", buckets)
	}
	value := minuteAverage(buckets[historyBucket{firstHour, "v"}])
	if value == nil || *value != 450 {
		t.Fatalf("hourly mean = %v, expected mean of available valid minutes to be 450", value)
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
