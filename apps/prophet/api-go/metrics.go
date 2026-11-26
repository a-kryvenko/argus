package main

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

const metricsPath = "/internal/v1/public-forecast-metrics/"

type metricReports struct {
	workdir string
	models  modelRegistry
}

func (reports metricReports) serve(w http.ResponseWriter, r *http.Request) {
	target := r.PathValue("product")
	product, ok := publicforecast.Products[target]
	if !ok {
		detail(w, 404, "Unknown forecast product")
		return
	}
	report, err := reports.read(target, product)
	if err != nil {
		unavailable(w, "Metrics are not ready", err)
		return
	}
	writeJSON(w, 200, report)
}

func (reports metricReports) read(target string, product publicforecast.Product) (*publicforecast.ForecastMetrics, error) {
	var path string
	if target == "geomagnetic-activity" {
		path = reports.models["geomagnetic_ordinal"].MetricsReport
	}
	for _, variable := range product.Variables {
		if path != "" {
			break
		}
		path = reports.models[variable.Artifact].MetricsReport
	}
	if path == "" {
		return nil, errors.New("metrics report is not configured")
	}
	raw, err := reports.readFile(path)
	if err != nil {
		return nil, err
	}
	report, err := metricsData(raw, target, product)
	if err != nil {
		return nil, err
	}
	if target == "geomagnetic-activity" {
		if fastPath := reports.models["geomagnetic_fast_l1"].MetricsReport; fastPath != "" {
			fast, err := reports.readFastL1(fastPath)
			if err != nil {
				log.Printf("read fast L1 metrics: %v", err)
			} else {
				report.FastL1Evaluation = fast
			}
		}
	}
	return report, nil
}

func (reports metricReports) readFile(path string) ([]byte, error) {
	if !filepath.IsAbs(path) {
		path = filepath.Join(reports.workdir, path)
	}
	return os.ReadFile(path)
}

func (reports metricReports) readFastL1(path string) (*publicforecast.FastL1Evaluation, error) {
	raw, err := reports.readFile(path)
	if err != nil {
		return nil, err
	}
	var fast publicforecast.FastL1Evaluation
	if err := json.Unmarshal(raw, &fast); err != nil {
		return nil, err
	}
	if fast.Period == "" || fast.Sample == "" || fast.Notes == nil || len(fast.ByDelay) == 0 {
		return nil, errors.New("incomplete fast L1 evaluation")
	}
	previous := -1
	for _, row := range fast.ByDelay {
		if row.DelayMinutes <= previous || row.Targets < 1 || row.Updates < 0 || row.Updates > row.Targets ||
			row.ObservedEvents < 0 || row.ObservedEvents > row.Targets ||
			!validFastL1Scores(row.Baseline, row) || !validFastL1Scores(row.Candidate, row) ||
			!validInterval(row.ApMAEDeltaCI95) || !validInterval(row.Kp5BrierDeltaCI95) {
			return nil, errors.New("invalid fast L1 comparison")
		}
		previous = row.DelayMinutes
	}
	return &fast, nil
}

func validFastL1Scores(scores *publicforecast.FastL1Scores, row publicforecast.FastL1Delay) bool {
	if scores == nil || scores.ApMAE < 0 || scores.ApIntervalWidth80 < 0 ||
		scores.Hits < 0 || scores.Hits > row.ObservedEvents ||
		scores.FalseAlerts < 0 || scores.FalseAlerts > row.Targets-row.ObservedEvents {
		return false
	}
	for _, value := range []float64{scores.Kp4Brier, scores.Kp5Brier, scores.Kp6Brier,
		scores.ApCoverage80, scores.AlertThreshold} {
		if value < 0 || value > 1 {
			return false
		}
	}
	return true
}

func validInterval(values []float64) bool {
	return values == nil || len(values) == 2 && values[0] <= values[1]
}

func metricsData(raw []byte, target string, product publicforecast.Product) (*publicforecast.ForecastMetrics, error) {
	var report publicforecast.ForecastMetrics
	if json.Unmarshal(raw, &report) != nil || report.Target != target || report.Variables == nil {
		return nil, errors.New("invalid evaluation report")
	}
	available := make([]string, 0, len(report.Variables))
	for name, variable := range report.Variables {
		available = append(available, name)
		if variable.Continuous != nil {
			if variable.Continuous.Quantiles == nil {
				return nil, errors.New("missing quantile metrics")
			}
			var err error
			variable.Continuous.ByLeadHour, err = trimMetrics(variable.Continuous.ByLeadHour, product.Horizon,
				func(point publicforecast.ContinuousMetricsPoint) *int { return point.LeadHours })
			if err != nil {
				return nil, err
			}
		}
		if variable.Binary == nil {
			variable.Binary = []publicforecast.BinaryMetricsSeries{}
		}
		for i := range variable.Binary {
			series := &variable.Binary[i]
			if series.Operator == "" {
				series.Operator = "gte"
			}
			if series.Threshold == nil || series.Operator != "gte" {
				return nil, errors.New("invalid threshold metrics")
			}
			var err error
			series.ByLeadHour, err = trimMetrics(series.ByLeadHour, product.Horizon,
				func(point publicforecast.BinaryMetricsPoint) *int { return point.LeadHours })
			if err != nil {
				return nil, err
			}
			for i := range series.ByLeadHour {
				if series.ByLeadHour[i].Reliability == nil {
					series.ByLeadHour[i].Reliability = []publicforecast.ReliabilityPoint{}
				}
			}
		}
		report.Variables[name] = variable
	}
	report.Meta = publicforecast.Metadata(product, available)
	return &report, nil
}

func trimMetrics[T any](points []T, horizon int, leadHours func(T) *int) ([]T, error) {
	if points == nil {
		return nil, errors.New("missing metrics by lead hour")
	}
	kept := points[:0]
	for _, point := range points {
		lead := leadHours(point)
		if lead == nil {
			return nil, errors.New("invalid metric lead hour")
		}
		if *lead <= horizon {
			kept = append(kept, point)
		}
	}
	return kept, nil
}
