package publicforecast

import (
	"encoding/json"
	"reflect"
	"time"

	"github.com/danielgtaylor/huma/v2"
)

type Published[T any] struct{ raw json.RawMessage }

func (p Published[T]) MarshalJSON() ([]byte, error) { return p.raw.MarshalJSON() }

func (Published[T]) Schema(registry huma.Registry) *huma.Schema {
	return registry.Schema(reflect.TypeFor[T](), true, "")
}

type CMESpeedOutlook struct {
	ModelVersion        string           `json:"model_version"`
	IntegratedIntoSpeed bool             `json:"integrated_into_speed" required:"false" default:"false"`
	AsOf                *time.Time       `json:"as_of" required:"false" nullable:"true"`
	State               string           `json:"state" enum:"available,partial,unavailable"`
	ConditionalOnImpact bool             `json:"conditional_on_impact" required:"false" default:"true" enum:"true"`
	Units               string           `json:"units" required:"false" default:"km/s" enum:"km/s"`
	ScenarioKind        string           `json:"scenario_kind" required:"false" default:"conditional_scenarios_not_quantiles" enum:"conditional_scenarios_not_quantiles"`
	BackgroundModel     map[string]any   `json:"background_model" required:"false"`
	Events              []map[string]any `json:"events" required:"false"`
	ExcludedEvents      []map[string]any `json:"excluded_events" required:"false"`
	Points              []CMESpeedPoint  `json:"points" required:"false"`
	Reason              *string          `json:"reason" required:"false" nullable:"true"`
	ModelSHA256         *string          `json:"model_sha256" required:"false" nullable:"true" pattern:"^[0-9a-f]{64}$"`
}

type CMESpeedPoint struct {
	ValidTime       time.Time `json:"valid_time"`
	SpeedBackground float64   `json:"speed_background" minimum:"0.0"`
	SpeedNominal    float64   `json:"speed_nominal" minimum:"0.0"`
	SpeedEarly      float64   `json:"speed_early" minimum:"0.0"`
	SpeedLate       float64   `json:"speed_late" minimum:"0.0"`
	ScenarioLower   float64   `json:"scenario_lower" minimum:"0.0"`
	ScenarioUpper   float64   `json:"scenario_upper" minimum:"0.0"`
	SpeedIntegrated *float64  `json:"speed_integrated" required:"false" nullable:"true" minimum:"0.0"`
}
