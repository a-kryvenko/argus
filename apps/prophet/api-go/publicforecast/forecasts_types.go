package publicforecast

import "time"

type BinaryForecast struct {
	Threshold   float64 `json:"threshold" doc:"Inclusive threshold in the parent variable's units."`
	Probability float64 `json:"probability" doc:"Probability of value >= threshold, from 0 to 1 (not percent)."`
}

type BinaryMetricsPoint struct {
	LeadHours        *int               `nullable:"false" json:"lead_hours" doc:"Number of hours between forecast issuance (issue_time) and the time being predicted (valid_time)."`
	BrierScore       *float64           `json:"brier_score" required:"false" nullable:"true" doc:"Mean squared error of event probabilities; lower is better. Null when unavailable."`
	RocAuc           *float64           `json:"roc_auc" required:"false" nullable:"true" doc:"Area under the ROC curve; higher is better. Null when unavailable."`
	AveragePrecision *float64           `json:"average_precision" required:"false" nullable:"true" doc:"Average precision summarizing precision and recall; higher is better. Null when unavailable."`
	ThreatScore      *float64           `json:"threat_score" required:"false" nullable:"true" doc:"Hits divided by hits + misses + false alarms (critical success index). Null when unavailable."`
	HeidkeSkillScore *float64           `json:"heidke_skill_score" required:"false" nullable:"true" doc:"Classification skill relative to chance agreement; 1 is perfect, 0 means no skill over chance. Null when unavailable."`
	Reliability      []ReliabilityPoint `json:"reliability" required:"false" doc:"Calibration points comparing predicted probabilities with observed frequencies; empty when unavailable."`
}

type BinaryMetricsSeries struct {
	Threshold  *float64             `nullable:"false" json:"threshold" doc:"Event threshold in the variable's units; see the endpoint's product table."`
	Operator   string               `json:"operator" required:"false" default:"gte" enum:"gte" doc:"gte means the event is variable value >= threshold."`
	ByLeadHour []BinaryMetricsPoint `json:"by_lead_hour" doc:"Threshold metrics grouped by forecast lead time in hours."`
}

type ContinuousMetrics struct {
	Quantiles  []float64                `json:"quantiles" doc:"Evaluated quantile levels as fractions, corresponding to q10, q50 and q90 in the forecast."`
	ByLeadHour []ContinuousMetricsPoint `json:"by_lead_hour" doc:"Quantile/regression metrics grouped by forecast lead time in hours."`
}

type ContinuousMetricsPoint struct {
	LeadHours *int           `nullable:"false" json:"lead_hours" doc:"Number of hours between forecast issuance (issue_time) and the time being predicted (valid_time)."`
	Values    map[string]any `json:"values" doc:"Metric names mapped to numeric scores, or null for missing scores. Available names depend on the product's evaluation output."`
}

type EvaluationContext struct {
	Label  string   `json:"label"`
	Period string   `json:"period"`
	Sample string   `json:"sample"`
	Notes  []string `json:"notes" required:"false"`
}

type Forecast struct {
	Target             string                      `json:"target" doc:"Forecast product identifier; see the endpoint's target list."`
	IssueTime          time.Time                   `json:"issue_time" doc:"Date and time when the forecast was made (ISO 8601)."`
	HorizonHours       int                         `json:"horizon_hours" doc:"Largest lead_hours among returned predictions; may be shorter than the product's maximum horizon."`
	AvailableVariables []string                    `json:"available_variables" doc:"Variable names available for this forecast issuance. A particular prediction may contain only a subset."`
	Predictions        []*ForecastPoint            `json:"predictions" doc:"Forecast points, each with its own valid_time and lead_hours."`
	Meta               *ForecastMetadata           `json:"meta,omitempty" required:"false" allowNull:"true" doc:"Additional descriptions, included only with meta=true."`
	CMESpeed           *Published[CMESpeedOutlook] `json:"cme_speed" required:"false" allowNull:"true" doc:"CME-conditioned speed scenarios from this release; separate from background quantiles and threshold probabilities."`
}

type ForecastMetadata struct {
	Variables map[string]ForecastVariableMetadata `json:"variables"`
}

type ForecastMetrics struct {
	Evaluation *EvaluationContext         `json:"evaluation" required:"false" allowNull:"true"`
	Target     string                     `json:"target" doc:"Forecast product identifier; see the endpoint's target list."`
	Variables  map[string]VariableMetrics `json:"variables" doc:"Metrics keyed by variable name. See the endpoint's product table for names and units. Variables without available metrics are omitted."`
	Meta       *ForecastMetadata          `json:"meta,omitempty" required:"false" allowNull:"true" doc:"Additional descriptions, included only with meta=true."`
}

type ForecastPoint struct {
	Valid     time.Time                    `json:"valid_time" doc:"Date and time for which the forecast predicts values (ISO 8601)."`
	Lead      int                          `json:"lead_hours" doc:"Number of hours between forecast issuance (issue_time) and the time being predicted (valid_time)."`
	Variables map[string]*VariableForecast `json:"variables" doc:"Values keyed by variable name. See the endpoint's product table for the variables and units supported by each target. Only available variables are included; individual forecast points may contain a subset."`
}

type ForecastVariableMetadata struct {
	Unit string `json:"unit"`
}

type QuantileForecast struct {
	Q10 float64 `json:"q10" doc:"Predicted 10th percentile (quantile 0.1)."`
	Q50 float64 `json:"q50" doc:"Predicted median (quantile 0.5)."`
	Q90 float64 `json:"q90" doc:"Predicted 90th percentile (quantile 0.9)."`
}

type ReliabilityPoint struct {
	PredictedProbability float64 `json:"predicted_probability" doc:"Predicted event probability for this calibration point, from 0 to 1."`
	ObservedFrequency    float64 `json:"observed_frequency" doc:"Observed fraction of events for this calibration point, from 0 to 1."`
}

type VariableForecast struct {
	Continuous *QuantileForecast `json:"continuous" required:"false" allowNull:"true" doc:"Quantile forecast, or null when unavailable or unsupported for this variable."`
	Binary     []BinaryForecast  `json:"binary" required:"false" doc:"Available threshold probabilities; empty when unavailable or unsupported. Thresholds with missing probabilities are omitted."`
}

type VariableMetrics struct {
	Continuous *ContinuousMetrics    `json:"continuous" required:"false" allowNull:"true" doc:"Quantile/regression metrics, or null when unavailable or unsupported."`
	Binary     []BinaryMetricsSeries `json:"binary" required:"false" doc:"Metrics for available thresholds; empty when unavailable or unsupported."`
}
