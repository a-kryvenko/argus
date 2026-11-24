package publicforecast

import "time"

type ArrivalSourceReceipt struct {
	PollID     string    `json:"poll_id" format:"uuid"`
	SHA256     string    `json:"sha256" pattern:"^[0-9a-f]{64}$"`
	ReceivedAt time.Time `json:"received_at"`
}

type CMEEnsembleForecast struct {
	ForecastID                string             `json:"forecast_id"`
	ModelVersion              string             `json:"model_version" required:"false" default:"argus-dbem-v1" enum:"argus-dbem-v1"`
	CMEID                     string             `json:"cme_id"`
	SourcePollID              string             `json:"source_poll_id" format:"uuid"`
	SourceSHA256              string             `json:"source_sha256" pattern:"^[0-9a-f]{64}$"`
	AnalysisSHA256            string             `json:"analysis_sha256" pattern:"^[0-9a-f]{64}$"`
	AvailableAt               time.Time          `json:"available_at"`
	InitialTime               time.Time          `json:"initial_time"`
	InitialParameters         map[string]float64 `json:"initial_parameters"`
	Policy                    map[string]any     `json:"policy"`
	Seed                      int                `json:"seed" minimum:"0.0"`
	EnsembleSize              int                `json:"ensemble_size" exclusiveMinimum:"0.0"`
	GeometricHitCount         int                `json:"geometric_hit_count" minimum:"0.0"`
	GeometricHitFraction      float64            `json:"geometric_hit_fraction" minimum:"0.0" maximum:"1.0"`
	ArrivalCount              int                `json:"arrival_count" minimum:"0.0"`
	RightCensoredCount        int                `json:"right_censored_count" minimum:"0.0"`
	FutureArrivalCount        int                `json:"future_arrival_count" minimum:"0.0"`
	ArrivalWithinHorizonCount int                `json:"arrival_within_horizon_count" minimum:"0.0"`
	Status                    string             `json:"status" enum:"geometric_miss,right_censored,expected,elapsed,timing_straddles_issue"`
	ArrivalFeature            string             `json:"arrival_feature" required:"false" default:"cme_leading_edge" enum:"cme_leading_edge"`
	QuantileKind              string             `json:"quantile_kind" required:"false" default:"uncalibrated_conditional_ensemble" enum:"uncalibrated_conditional_ensemble"`
	Calibrated                bool               `json:"calibrated" required:"false" default:"false" enum:"false"`
	ArrivalQ10                *time.Time         `json:"arrival_q10" required:"false" nullable:"true"`
	ArrivalQ50                *time.Time         `json:"arrival_q50" required:"false" nullable:"true"`
	ArrivalQ90                *time.Time         `json:"arrival_q90" required:"false" nullable:"true"`
	SpeedQ10KmS               *float64           `json:"speed_q10_km_s" required:"false" nullable:"true" exclusiveMinimum:"0.0"`
	SpeedQ50KmS               *float64           `json:"speed_q50_km_s" required:"false" nullable:"true" exclusiveMinimum:"0.0"`
	SpeedQ90KmS               *float64           `json:"speed_q90_km_s" required:"false" nullable:"true" exclusiveMinimum:"0.0"`
	EarthDistanceAtMedianKm   *float64           `json:"earth_distance_at_median_km" required:"false" nullable:"true" exclusiveMinimum:"0.0"`
	PotentialInteractionIds   []string           `json:"potential_interaction_ids" required:"false"`
}

type CoronalHoleFrame struct {
	FeatureVersion string              `json:"feature_version" required:"false" default:"euv-dark-unipolar-v1" enum:"euv-dark-unipolar-v1"`
	ObservedAt     time.Time           `json:"observed_at"`
	AvailableAt    time.Time           `json:"available_at"`
	B0Deg          float64             `json:"b0_deg"`
	SourceReceipts []map[string]any    `json:"source_receipts"`
	Regions        []CoronalHoleRegion `json:"regions"`
	MagneticState  string              `json:"magnetic_state" enum:"available,unavailable,rejected"`
}

type CoronalHoleRegion struct {
	RegionID             string   `json:"region_id"`
	LongitudeDeg         float64  `json:"longitude_deg" minimum:"-60.0" maximum:"60.0"`
	WestEdgeDeg          float64  `json:"west_edge_deg" minimum:"-60.0" maximum:"60.0"`
	EastEdgeDeg          float64  `json:"east_edge_deg" minimum:"-60.0" maximum:"60.0"`
	LatitudeDeg          float64  `json:"latitude_deg" minimum:"-60.0" maximum:"60.0"`
	AreaFraction         float64  `json:"area_fraction" maximum:"1.0" exclusiveMinimum:"0.0"`
	EarthBandFraction    float64  `json:"earth_band_fraction" minimum:"0.0" maximum:"1.0"`
	MagneticUnipolarity  *float64 `json:"magnetic_unipolarity" required:"false" nullable:"true" minimum:"0.0" maximum:"1.0"`
	MeanRadialFieldGauss *float64 `json:"mean_radial_field_gauss" required:"false" nullable:"true"`
	Classification       string   `json:"classification" enum:"euv_dark_region,magnetically_supported_candidate"`
}

type HSSForecast struct {
	ForecastID           string     `json:"forecast_id"`
	Kind                 string     `json:"kind" required:"false" default:"hss_candidate" enum:"hss_candidate"`
	Status               string     `json:"status" enum:"expected,ongoing_candidate"`
	TimingMethod         string     `json:"timing_method" enum:"proswin_profile,rotation_profile,coronal_hole_transport"`
	ArrivalAt            time.Time  `json:"arrival_at"`
	ArrivalWindowStart   time.Time  `json:"arrival_window_start"`
	ArrivalWindowEnd     time.Time  `json:"arrival_window_end"`
	WindowKind           string     `json:"window_kind" enum:"uncalibrated_timing_envelope,heuristic_transport_range"`
	PeakAt               *time.Time `json:"peak_at" required:"false" nullable:"true"`
	PeakSpeedKmS         *float64   `json:"peak_speed_km_s" required:"false" nullable:"true" exclusiveMinimum:"0.0" exclusiveMaximum:"3000.0"`
	EndAt                *time.Time `json:"end_at" required:"false" nullable:"true"`
	OnsetLeftCensored    bool       `json:"onset_left_censored" required:"false" default:"false"`
	EndRightCensored     bool       `json:"end_right_censored" required:"false" default:"false"`
	EvidenceSources      []string   `json:"evidence_sources"`
	CoronalHoleRegionIds []string   `json:"coronal_hole_region_ids" required:"false"`
	MagneticSupport      bool       `json:"magnetic_support" required:"false" default:"false"`
	RecurrenceSupport    bool       `json:"recurrence_support" required:"false" default:"false"`
	CMEOverlapIds        []string   `json:"cme_overlap_ids" required:"false"`
	Attribution          string     `json:"attribution" required:"false" default:"candidate" enum:"candidate,ambiguous_cme_overlap"`
}

type HSSSpeedPoint struct {
	ValidTime time.Time `json:"valid_time"`
	SpeedKmS  float64   `json:"speed_km_s" exclusiveMinimum:"0.0" exclusiveMaximum:"3000.0"`
	Source    string    `json:"source" enum:"PROSWIN,rotation"`
}

type L1Detection struct {
	DetectionID       string             `json:"detection_id"`
	Kind              string             `json:"kind" required:"false" default:"compression_candidate" enum:"compression_candidate"`
	OnsetAt           time.Time          `json:"onset_at"`
	DetectedAt        time.Time          `json:"detected_at"`
	Evidence          map[string]float64 `json:"evidence"`
	CandidateEventIds []string           `json:"candidate_event_ids" required:"false"`
	Attribution       string             `json:"attribution" required:"false" default:"unresolved" enum:"unresolved"`
}

type SolarWindEvent struct {
	EventID            string                 `json:"event_id"`
	Kind               string                 `json:"kind" enum:"cme,hss,interplanetary_shock,solar_wind_disturbance"`
	Status             string                 `json:"status" enum:"detected,predicted_hit,no_predicted_hit,prediction_elapsed,observed"`
	Source             string                 `json:"source" enum:"DONKI,DONKI/WSA-ENLIL,ARGUS"`
	SourceIds          []string               `json:"source_ids" required:"false"`
	SourceURL          *string                `json:"source_url" required:"false" nullable:"true"`
	AvailableAt        time.Time              `json:"available_at"`
	ObservedAt         *time.Time             `json:"observed_at" required:"false" nullable:"true"`
	ModelCompletedAt   *time.Time             `json:"model_completed_at" required:"false" nullable:"true"`
	ArrivalAt          *time.Time             `json:"arrival_at" required:"false" nullable:"true"`
	ArrivalTarget      *string                `json:"arrival_target" required:"false" nullable:"true" enum:"Earth"`
	DurationHours      *float64               `json:"duration_hours" required:"false" nullable:"true" exclusiveMinimum:"0.0"`
	GlancingBlow       *bool                  `json:"glancing_blow" required:"false" nullable:"true"`
	MinorImpact        *bool                  `json:"minor_impact" required:"false" nullable:"true"`
	SimulationID       *string                `json:"simulation_id" required:"false" nullable:"true"`
	InitialSpeedKmS    *float64               `json:"initial_speed_km_s" required:"false" nullable:"true" exclusiveMinimum:"0.0"`
	ArrivalModel       *string                `json:"arrival_model" required:"false" nullable:"true"`
	ArrivalMethod      *string                `json:"arrival_method" required:"false" nullable:"true"`
	ArrivalReceipts    []ArrivalSourceReceipt `json:"arrival_receipts" required:"false"`
	ArrivalWindowStart *time.Time             `json:"arrival_window_start" required:"false" nullable:"true"`
	ArrivalWindowEnd   *time.Time             `json:"arrival_window_end" required:"false" nullable:"true"`
}

type SolarWindEventForecast struct {
	SchemaVersion   int                       `json:"schema_version" required:"false" default:"1" enum:"1"`
	Product         string                    `json:"product" required:"false" default:"solar-wind-events" enum:"solar-wind-events"`
	ModelVersion    string                    `json:"model_version" required:"false" default:"solar-wind-events-v1" enum:"solar-wind-events-v1,solar-wind-events-v2,solar-wind-events-v3,solar-wind-events-v4"`
	IssueTime       time.Time                 `json:"issue_time"`
	HorizonHours    int                       `json:"horizon_hours" required:"false" default:"168" enum:"168"`
	Experimental    bool                      `json:"experimental" required:"false" default:"true" enum:"true"`
	SourceStatus    map[string]map[string]any `json:"source_status"`
	Events          []SolarWindEvent          `json:"events"`
	L1Detections    []L1Detection             `json:"l1_detections"`
	HSSForecasts    []HSSForecast             `json:"hss_forecasts" required:"false"`
	HSSSpeedProfile []HSSSpeedPoint           `json:"hss_speed_profile" required:"false"`
	CoronalHoles    *CoronalHoleFrame         `json:"coronal_holes" required:"false" allowNull:"true"`
	CMEEnsembles    []CMEEnsembleForecast     `json:"cme_ensembles" required:"false"`
	CMESpeed        *CMESpeedOutlook          `json:"cme_speed" required:"false" allowNull:"true"`
	Limitations     []string                  `json:"limitations" required:"false"`
}
