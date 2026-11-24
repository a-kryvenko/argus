package main

import "time"

type ForecastVerification struct {
	Product  string              `json:"product"`
	Protocol string              `json:"protocol" required:"false" default:"published-hourly-v1" enum:"published-hourly-v1"`
	Days     int                 `json:"days" required:"false" default:"30"`
	Start    time.Time           `json:"start"`
	End      time.Time           `json:"end"`
	Groups   []VerificationGroup `json:"groups" required:"false"`
}

type VerificationGroup struct {
	Artifact      string             `json:"artifact"`
	Model         *string            `json:"model" required:"false" nullable:"true"`
	ModelSHA256   *string            `json:"model_sha256" required:"false" nullable:"true"`
	LastIssueTime *time.Time         `json:"last_issue_time" required:"false" nullable:"true"`
	Releases      int                `json:"releases"`
	EvaluatedAt   time.Time          `json:"evaluated_at"`
	Counts        map[string]int     `json:"counts"`
	ByLeadHour    []VerificationLead `json:"by_lead_hour" required:"false"`
}

type VerificationLead struct {
	LeadHours  int                            `json:"lead_hours"`
	Counts     map[string]int                 `json:"counts"`
	Continuous map[string]float64             `json:"continuous" required:"false" nullable:"true"`
	Binary     map[string]map[string]*float64 `json:"binary" required:"false"`
}
