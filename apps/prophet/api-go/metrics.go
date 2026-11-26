package main

import (
	"encoding/json"
	"errors"
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
	if !filepath.IsAbs(path) {
		path = filepath.Join(reports.workdir, path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return metricsData(raw, target, product)
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
