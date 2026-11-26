package main

import (
	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
	"time"
)

type DragMetadata struct {
	Model  string                         `json:"model" enum:"circular_leo_drag_v1"`
	Source publicforecast.DensityMetadata `json:"source"`
}

type DragPoint struct {
	ValidTime                     time.Time `json:"valid_time"`
	LeadHours                     int       `json:"lead_hours" minimum:"0.0" maximum:"48.0"`
	MeanDensityKgM3               float64   `json:"mean_density_kg_m3" exclusiveMinimum:"0.0"`
	MeanDragAccelMS2              float64   `json:"mean_drag_accel_m_s2" minimum:"0.0"`
	MeanAlongTrackDecelerationMS2 float64   `json:"mean_along_track_deceleration_m_s2" minimum:"0.0"`
	DeltaVLossMS                  float64   `json:"delta_v_loss_m_s" minimum:"0.0" doc:"Cumulative along-track drag impulse per unit mass."`
	EstimatedAltitudeLossM        float64   `json:"estimated_altitude_loss_m" minimum:"0.0"`
}

type DragSource struct {
	ReleaseID              string     `json:"release_id" format:"uuid"`
	IssueTime              time.Time  `json:"issue_time"`
	ObservedAt             time.Time  `json:"observed_at"`
	DTCObservedAt          *time.Time `json:"dtc_observed_at" required:"false" nullable:"true"`
	GeomagneticObservedAt  *time.Time `json:"geomagnetic_observed_at" required:"false" nullable:"true"`
	DriverMode             string     `json:"driver_mode" enum:"observed_persistence"`
	BackgroundInterpolated bool       `json:"background_interpolated"`
}

type DragThresholds struct {
	Elevated float64 `json:"elevated_altitude_loss_m" exclusiveMinimum:"0.0"`
	High     float64 `json:"high_altitude_loss_m" exclusiveMinimum:"0.0"`
}

type LeoDragAssessment struct {
	ComputedAt             time.Time      `json:"computed_at"`
	StartTime              time.Time      `json:"start_time"`
	EndTime                time.Time      `json:"end_time"`
	Inputs                 LeoDragRequest `json:"inputs"`
	Source                 DragSource     `json:"source"`
	MeanDensityKgM3        float64        `json:"mean_density_kg_m3"`
	MeanDragAccelMS2       float64        `json:"mean_drag_accel_m_s2"`
	DeltaVLossMS           float64        `json:"delta_v_loss_m_s"`
	EstimatedAltitudeLossM float64        `json:"estimated_altitude_loss_m"`
	DragRisk               string         `json:"drag_risk" enum:"not_assessed,low,elevated,high"`
	RiskReason             string         `json:"risk_reason"`
	Assumptions            []string       `json:"assumptions"`
	Predictions            []DragPoint    `json:"predictions"`
	Meta                   *DragMetadata  `json:"meta,omitempty" required:"false" allowNull:"true" doc:"Drag model and density inputs used for this assessment."`
}

type LeoDragRequest struct {
	Altitude    *float64        `nullable:"false" json:"altitude_km" minimum:"200.0" maximum:"800.0"`
	Inclination *float64        `nullable:"false" json:"inclination_deg" minimum:"0.0" maximum:"180.0"`
	Mass        *float64        `nullable:"false" json:"mass_kg" minimum:"0.01" maximum:"10000000.0"`
	Area        *float64        `nullable:"false" json:"effective_area_m2" maximum:"1000000.0" exclusiveMinimum:"0.0"`
	Coefficient *float64        `nullable:"false" json:"drag_coefficient" maximum:"10.0" exclusiveMinimum:"0.0"`
	Horizon     int             `json:"horizon_hours" required:"false" enum:"24,48"`
	Thresholds  *DragThresholds `json:"thresholds" required:"false" allowNull:"true" doc:"User-defined cumulative altitude-loss thresholds over the requested horizon. Without thresholds, drag_risk is not_assessed."`
}
