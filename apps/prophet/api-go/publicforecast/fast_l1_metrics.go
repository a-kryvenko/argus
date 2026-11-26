package publicforecast

type FastL1Evaluation struct {
	Period  string        `json:"period" doc:"Period of the archived verification targets."`
	Sample  string        `json:"sample" doc:"Matched target intervals and evaluation horizon."`
	Notes   []string      `json:"notes" doc:"Replay timing and alert threshold assumptions."`
	ByDelay []FastL1Delay `json:"by_delay" doc:"Pairwise scores under each simulated observation delay."`
}

type FastL1Delay struct {
	DelayMinutes      int           `json:"delay_minutes" doc:"Simulated L1 observation delivery delay in minutes."`
	Targets           int           `json:"targets" doc:"Number of matched native three-hour target intervals."`
	Updates           int           `json:"updates" doc:"Number of intervals updated by the L1 model."`
	ObservedEvents    int           `json:"observed_events" doc:"Number of target intervals reaching Kp 5."`
	Baseline          *FastL1Scores `json:"baseline" nullable:"false" doc:"Scheduled Kp/ap forecast on the same targets."`
	Candidate         *FastL1Scores `json:"candidate" nullable:"false" doc:"Fast L1 forecast with scheduled fallback."`
	ApMAEDeltaCI95    []float64     `json:"ap_mae_delta_ci95,omitempty" required:"false" doc:"Paired 95% interval for candidate minus baseline ap MAE."`
	Kp5BrierDeltaCI95 []float64     `json:"kp5_brier_delta_ci95,omitempty" required:"false" doc:"Paired 95% interval for candidate minus baseline Kp ≥ 5 Brier score."`
}

type FastL1Scores struct {
	Kp4Brier          float64 `json:"kp4_brier"`
	Kp5Brier          float64 `json:"kp5_brier"`
	Kp6Brier          float64 `json:"kp6_brier"`
	ApMAE             float64 `json:"ap_mae"`
	ApCoverage80      float64 `json:"ap_coverage_80" doc:"Observed fraction covered by the Q10–Q90 interval."`
	ApIntervalWidth80 float64 `json:"ap_interval_width_80" doc:"Mean width of the Q10–Q90 interval in ap units."`
	AlertThreshold    float64 `json:"alert_threshold" doc:"Evaluation cutoff for P(Kp ≥ 5), expressed as a fraction."`
	Hits              int     `json:"hits" doc:"Observed Kp ≥ 5 intervals with an alert."`
	FalseAlerts       int     `json:"false_alerts" doc:"Observed Kp < 5 intervals with an alert."`
}
