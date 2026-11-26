package main

import (
	"encoding/json"
	"errors"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

func (reports metricReports) readFastBt(path string) (*publicforecast.FastBtEvaluation, error) {
	raw, err := reports.readFile(path)
	if err != nil {
		return nil, err
	}
	var result publicforecast.FastBtEvaluation
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, err
	}
	if result.Period == "" || result.Sample == "" || result.Notes == nil || len(result.Cohorts) == 0 {
		return nil, errors.New("incomplete fast Bt evaluation")
	}
	seen := make(map[string]bool, len(result.Cohorts))
	for _, cohort := range result.Cohorts {
		if cohort.ID == "" || cohort.Label == "" || seen[cohort.ID] || len(cohort.ByDelay) == 0 {
			return nil, errors.New("invalid fast Bt cohort")
		}
		seen[cohort.ID] = true
		previous := -1
		for _, row := range cohort.ByDelay {
			if row.DelayMinutes <= previous || !validFastBtDelay(row) {
				return nil, errors.New("invalid fast Bt comparison")
			}
			previous = row.DelayMinutes
		}
	}
	return &result, nil
}

func validFastBtDelay(row publicforecast.FastBtDelay) bool {
	if row.Targets < 1 || row.Updates < 0 || row.Updates > row.Targets || len(row.Thresholds) != 3 {
		return false
	}
	previous := row.Targets
	for index, metric := range row.Thresholds {
		if metric.Threshold != (index+1)*5 || metric.ObservedEvents < 0 || metric.ObservedEvents > previous ||
			!validFastBtScores(metric.Baseline, metric.ObservedEvents, row.Targets) ||
			!validFastBtScores(metric.Candidate, metric.ObservedEvents, row.Targets) ||
			!validInterval(metric.BrierDeltaCI95) {
			return false
		}
		previous = metric.ObservedEvents
	}
	return true
}

func validFastBtScores(scores *publicforecast.FastBtScores, events, targets int) bool {
	if scores == nil || scores.BrierScore == nil || scores.AlertThreshold == nil ||
		scores.Hits < 0 || scores.Hits > events || scores.FalseAlerts < 0 || scores.FalseAlerts > targets-events {
		return false
	}
	values := []*float64{scores.BrierScore, scores.AlertThreshold, scores.AveragePrecision}
	for _, value := range values {
		if value != nil && (*value < 0 || *value > 1) {
			return false
		}
	}
	return true
}
