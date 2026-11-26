package main

import (
	"encoding/json"
	"time"

	"github.com/danielgtaylor/huma/v2"
)

type Coverage struct {
	ExpectedSlots     int       `json:"expected_slots"`
	UsableSlots       int       `json:"usable_slots"`
	MissingSlots      int       `json:"missing_slots"`
	InvalidSlots      int       `json:"invalid_slots"`
	Percent           *float64  `json:"percent" nullable:"true"`
	ResolutionSeconds int       `json:"resolution_seconds"`
	EvaluatedTo       time.Time `json:"evaluated_to"`
	Basis             string    `json:"basis" enum:"overlapping_intervals,sample_timestamps,available_aggregate_windows"`
	Gaps              []Gap     `json:"gaps"`
}

type CurrentIndexSample struct {
	Provenance
	TimeBasis     string    `json:"time_basis"`
	IntervalStart time.Time `json:"interval_start"`
	IntervalEnd   time.Time `json:"interval_end"`
	Value         *float64  `json:"value" nullable:"true"`
	Quality       *string   `json:"quality,omitempty" required:"false" nullable:"true" enum:"missing,flagged"`
}

type CurrentObservations struct {
	// A selected metric with no sample is null. A metric the caller did not select is omitted.
	Bx           **CurrentSolarSample `json:"bx,omitempty" required:"false" allowNull:"true"`
	By           **CurrentSolarSample `json:"by,omitempty" required:"false" allowNull:"true"`
	Bz           **CurrentSolarSample `json:"bz,omitempty" required:"false" allowNull:"true"`
	Bt           **CurrentSolarSample `json:"bt,omitempty" required:"false" allowNull:"true"`
	V            **CurrentSolarSample `json:"v,omitempty" required:"false" allowNull:"true"`
	N            **CurrentSolarSample `json:"n,omitempty" required:"false" allowNull:"true"`
	T            **CurrentSolarSample `json:"t,omitempty" required:"false" allowNull:"true"`
	Kp           **CurrentIndexSample `json:"kp,omitempty" required:"false" allowNull:"true"`
	Dst          **CurrentIndexSample `json:"dst,omitempty" required:"false" allowNull:"true"`
	AP           **CurrentSolarSample `json:"ap,omitempty" required:"false" allowNull:"true"`
	F107         **CurrentSolarSample `json:"f10_7,omitempty" required:"false" allowNull:"true"`
	S10          **CurrentSolarSample `json:"s10,omitempty" required:"false" allowNull:"true"`
	M10          **CurrentSolarSample `json:"m10,omitempty" required:"false" allowNull:"true"`
	Y10          **CurrentSolarSample `json:"y10,omitempty" required:"false" allowNull:"true"`
	XRSA         **CurrentSolarSample `json:"xrs_a,omitempty" required:"false" allowNull:"true"`
	XRSB         **CurrentSolarSample `json:"xrs_b,omitempty" required:"false" allowNull:"true"`
	Proton10MeV  **CurrentSolarSample `json:"proton_10mev,omitempty" required:"false" allowNull:"true"`
	Proton100MeV **CurrentSolarSample `json:"proton_100mev,omitempty" required:"false" allowNull:"true"`
	Changes1h    map[string]*float64  `json:"changes_1h,omitempty" required:"false" nullable:"true"`
}

type CurrentSolarSample struct {
	Provenance
	TimeBasis   string     `json:"time_basis"`
	ObservedAt  time.Time  `json:"observed_at"`
	IntervalEnd *time.Time `json:"interval_end,omitempty" required:"false" nullable:"true"`
	Value       *float64   `json:"value" nullable:"true"`
	Quality     *string    `json:"quality,omitempty" required:"false" nullable:"true" enum:"missing,flagged"`
}

type Gap struct {
	From   time.Time `json:"from"`
	To     time.Time `json:"to"`
	Reason string    `json:"reason" enum:"missing,invalid,partial"`
	Slots  int       `json:"slots"`
}

type IndexHistorySeries struct {
	SeriesContext
	Points     []IndexSample `json:"points"`
	Coverage   Coverage      `json:"coverage"`
	DataStatus string        `json:"data_status" enum:"estimated,realtime"`
}

type IndexSample struct {
	Provenance
	IntervalStart  time.Time `json:"interval_start"`
	IntervalEnd    time.Time `json:"interval_end"`
	IntervalStatus string    `json:"interval_status" enum:"in_progress,completed"`
	Value          *float64  `json:"value" nullable:"true"`
	Quality        *string   `json:"quality,omitempty" required:"false" nullable:"true" enum:"missing,flagged"`
}

type ObservationHistory struct {
	From              time.Time                `json:"from"`
	To                time.Time                `json:"to"`
	Series            map[string]HistorySeries `json:"series"`
	TimeBasis         *string                  `json:"time_basis,omitempty" required:"false" enum:"l1,near_earth"`
	ResolutionSeconds *int                     `json:"resolution_seconds,omitempty" required:"false" nullable:"true"`
	EvaluatedFrom     *time.Time               `json:"evaluated_from,omitempty" required:"false" nullable:"true"`
	EvaluatedTo       *time.Time               `json:"evaluated_to,omitempty" required:"false" nullable:"true"`
	Points            *[]map[string]any        `json:"points,omitempty" required:"false" nullable:"true"`
}

type Processing struct {
	UnavailableBuckets          int `json:"unavailable_buckets"`
	RecalculationPendingBuckets int `json:"recalculation_pending_buckets"`
	AvailableBuckets            int `json:"available_buckets"`
	ExpectedBuckets             int `json:"expected_buckets"`
}

type SolarHistorySeries struct {
	SeriesContext
	Points     []SolarSample `json:"points"`
	Coverage   Coverage      `json:"coverage"`
	Processing *Processing   `json:"processing,omitempty" required:"false" allowNull:"true"`
}

type SolarSample struct {
	Provenance
	ObservedAt           time.Time  `json:"observed_at"`
	Value                *float64   `json:"value" nullable:"true"`
	Quality              *string    `json:"quality,omitempty" required:"false" nullable:"true" enum:"missing,flagged"`
	IntervalEnd          *time.Time `json:"interval_end,omitempty" required:"false" nullable:"true"`
	Min                  *float64   `json:"min,omitempty" required:"false" nullable:"true"`
	Max                  *float64   `json:"max,omitempty" required:"false" nullable:"true"`
	Count                *int       `json:"count,omitempty" required:"false" nullable:"true"`
	ExpectedCount        *int       `json:"expected_count,omitempty" required:"false" nullable:"true"`
	CoveragePercent      *float64   `json:"coverage_percent,omitempty" required:"false" nullable:"true"`
	RecalculationPending *bool      `json:"recalculation_pending,omitempty" required:"false" nullable:"true"`
}

type HistorySeries struct {
	Solar *SolarHistorySeries
	Index *IndexHistorySeries
}

func (series HistorySeries) MarshalJSON() ([]byte, error) {
	if series.Index != nil {
		return json.Marshal(series.Index)
	}
	return json.Marshal(series.Solar)
}

func (HistorySeries) Schema(registry huma.Registry) *huma.Schema {
	return &huma.Schema{AnyOf: []*huma.Schema{schemaFor[SolarHistorySeries](registry), schemaFor[IndexHistorySeries](registry)}}
}

type Provenance struct {
	SourceProduct string     `json:"source_product,omitempty" required:"false"`
	Spacecraft    string     `json:"spacecraft,omitempty" required:"false"`
	ReceivedAt    *time.Time `json:"received_at,omitempty" required:"false" nullable:"true"`
}

type SeriesContext struct {
	TimeBasis         string `json:"time_basis"`
	ResolutionSeconds int    `json:"resolution_seconds"`
	Aggregation       string `json:"aggregation"`
}
