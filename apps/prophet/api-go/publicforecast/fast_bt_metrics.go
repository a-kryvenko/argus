package publicforecast

type FastBtEvaluation struct {
	Period  string         `json:"period" doc:"Period of the archived verification targets."`
	Sample  string         `json:"sample" doc:"Matched target clock and verification horizon."`
	Notes   []string       `json:"notes" doc:"Replay timing and alert cutoff assumptions."`
	Cohorts []FastBtCohort `json:"cohorts" doc:"All archived detections and the strict catalog-matched subset."`
}

type FastBtCohort struct {
	ID      string        `json:"id"`
	Label   string        `json:"label"`
	ByDelay []FastBtDelay `json:"by_delay"`
}

type FastBtDelay struct {
	DelayMinutes int               `json:"delay_minutes" doc:"Simulated L1 delivery delay in minutes."`
	Targets      int               `json:"targets" doc:"Number of matched complete native-L1 hours."`
	Updates      int               `json:"updates" doc:"Number of target hours updated by the fast Bt model."`
	Thresholds   []FastBtThreshold `json:"thresholds"`
}

type FastBtThreshold struct {
	Threshold      int           `json:"threshold" doc:"Inclusive Bt threshold in nT."`
	ObservedEvents int           `json:"observed_events" doc:"Number of target hours reaching this threshold."`
	Baseline       *FastBtScores `json:"baseline" nullable:"false"`
	Candidate      *FastBtScores `json:"candidate" nullable:"false"`
	BrierDeltaCI95 []float64     `json:"brier_delta_ci95,omitempty" required:"false" doc:"Paired 95% interval for candidate minus baseline Brier score."`
}

type FastBtScores struct {
	BrierScore       *float64 `json:"brier_score" nullable:"false"`
	AveragePrecision *float64 `json:"average_precision" nullable:"true"`
	AlertThreshold   *float64 `json:"alert_threshold" nullable:"false" doc:"Evaluation probability cutoff, expressed as a fraction."`
	Hits             int      `json:"hits" doc:"Observed exceedance hours with an alert."`
	FalseAlerts      int      `json:"false_alerts" doc:"Observed non-exceedance hours with an alert."`
}
