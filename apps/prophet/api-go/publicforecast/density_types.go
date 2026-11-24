package publicforecast

import (
	"time"

	"github.com/danielgtaylor/huma/v2"
)

type CalendarDate string

func (CalendarDate) Schema(huma.Registry) *huma.Schema {
	return &huma.Schema{Type: "string", Format: "date"}
}

type DensityBackground struct {
	Method         string  `json:"method" enum:"jb2008_fixed_ap4_v1"`
	GeomagneticAP  float64 `json:"geomagnetic_ap" required:"false" default:"4.0" enum:"4.0"`
	DTCKelvin      float64 `json:"dtc_kelvin" minimum:"0.0"`
	SpatialSupport string  `json:"spatial_support" required:"false" default:"longitude_mean" enum:"longitude_mean"`
	SolarDrivers   string  `json:"solar_drivers" required:"false" default:"same_as_forecast" enum:"same_as_forecast"`
}

type DensityCell struct {
	AltitudeKm       float64  `json:"altitude_km" minimum:"90.0" maximum:"2500.0"`
	LatitudeDeg      float64  `json:"latitude_deg" minimum:"-90.0" maximum:"90.0"`
	RhoKgM3          float64  `json:"rho_kg_m3" exclusiveMinimum:"0.0"`
	RhoLonP10KgM3    float64  `json:"rho_lon_p10_kg_m3" exclusiveMinimum:"0.0"`
	RhoLonP90KgM3    float64  `json:"rho_lon_p90_kg_m3" exclusiveMinimum:"0.0"`
	RhoReferenceKgM3 *float64 `json:"rho_reference_kg_m3" required:"false" nullable:"true" exclusiveMinimum:"0.0"`
}

type DensityForecast struct {
	Target                 string             `json:"target" required:"false" default:"atmospheric-density" enum:"atmospheric-density"`
	DriverMode             string             `json:"driver_mode" required:"false" default:"observed_persistence" enum:"observed_persistence"`
	IssueTime              time.Time          `json:"issue_time"`
	ObservedAt             time.Time          `json:"observed_at"`
	DTCObservedAt          *time.Time         `json:"dtc_observed_at" required:"false" nullable:"true"`
	GeomagneticObservedAt  *time.Time         `json:"geomagnetic_observed_at" required:"false" nullable:"true"`
	BackgroundInterpolated bool               `json:"background_interpolated" doc:"True when background drivers include gap-filled daily values. Details are available with meta=true."`
	HorizonHours           int                `json:"horizon_hours"`
	Predictions            []DensityPoint     `json:"predictions"`
	Reference              *DensityBackground `json:"reference" required:"false" allowNull:"true"`
	Meta                   *DensityMetadata   `json:"meta,omitempty" required:"false" allowNull:"true" doc:"Additional descriptions, included only with meta=true."`
}

type DensityMetadata struct {
	Model                      string                    `json:"model" required:"false" default:"JB2008" enum:"JB2008,DTM2020"`
	HistoryStart               time.Time                 `json:"history_start"`
	BackgroundMethod           string                    `json:"background_method" enum:"trailing_81_daily_values,trailing_81_daily_values_linear_gapfill,published_81_day_mean_persistence"`
	BackgroundMeanObservedAt   map[string]time.Time      `json:"background_mean_observed_at" required:"false"`
	BackgroundInterpolatedDays map[string][]CalendarDate `json:"background_interpolated_days"`
	DTCMethod                  *string                   `json:"dtc_method" required:"false" nullable:"true" enum:"causal_dst_ap_v1"`
	GeomagneticMethod          *string                   `json:"geomagnetic_method" required:"false" nullable:"true" enum:"causal_dst_ap_v1,ap_to_kp_lag3h_mean24h_v1"`
}

type DensityPoint struct {
	Valid time.Time     `json:"valid_time"`
	Lead  int           `json:"lead_hours" minimum:"0.0" maximum:"48.0"`
	Cells []DensityCell `json:"cells"`
}
