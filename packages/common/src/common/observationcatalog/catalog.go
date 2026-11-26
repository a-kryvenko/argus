// Package observationcatalog defines shared numeric series and their wire format.
package observationcatalog

import (
	_ "embed"
	"encoding/json"
	"time"
)

//go:embed catalog.json
var definition []byte

type Metric struct {
	ID              string   `json:"id"`
	Label           string   `json:"label"`
	Unit            string   `json:"unit"`
	Group           string   `json:"group"`
	NativeSeconds   int      `json:"native_seconds"`
	SourceInterval  bool     `json:"source_interval,omitempty"`
	Signed          bool     `json:"signed,omitempty"`
	Max             *float64 `json:"max,omitempty"`
	Frame           string   `json:"frame,omitempty"`
	Products        []string `json:"products,omitempty"`
	ArchiveProducts []string `json:"archive_products,omitempty"`
}

type Catalog struct {
	Metrics     []Metric          `json:"metrics"`
	ProductURLs map[string]string `json:"product_urls"`
}

var catalog = func() Catalog {
	var result Catalog
	if err := json.Unmarshal(definition, &result); err != nil {
		panic(err)
	}
	return result
}()

var Metrics = catalog.Metrics
var ProductURLs = catalog.ProductURLs
var byID = func() map[string]Metric {
	result := make(map[string]Metric, len(Metrics))
	for _, metric := range Metrics {
		result[metric.ID] = metric
	}
	return result
}()

func IDs(group string) []string {
	var ids []string
	for _, metric := range Metrics {
		if metric.Group == group {
			ids = append(ids, metric.ID)
		}
	}
	return ids
}

func Find(id string) (Metric, bool) {
	metric, found := byID[id]
	return metric, found
}

type Point struct {
	ObservedAt    time.Time  `json:"observed_at"`
	Value         *float64   `json:"value"`
	SourceProduct string     `json:"source_product,omitempty"`
	Spacecraft    string     `json:"spacecraft,omitempty"`
	ReceivedAt    *time.Time `json:"received_at,omitempty"`
	Quality       string     `json:"quality,omitempty"`
	Count         *int       `json:"count,omitempty"`
	ExpectedCount *int       `json:"expected_count,omitempty"`
	Min           *float64   `json:"min,omitempty"`
	Max           *float64   `json:"max,omitempty"`
}

type Series struct {
	Points            []Point `json:"points"`
	ResolutionSeconds int     `json:"resolution_seconds"`
	TimeBasis         string  `json:"time_basis"`
	Aggregation       string  `json:"aggregation"`
}
