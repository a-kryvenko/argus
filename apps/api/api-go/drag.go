package main

import (
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"
)

func (a *api) leoDrag(w http.ResponseWriter, r *http.Request, respond func(Published[LeoDragAssessment]), validBody func(any) bool) {
	if contentType := r.Header.Get("Content-Type"); contentType != "" {
		mediaType, _, err := mime.ParseMediaType(contentType)
		if err != nil || (mediaType != "application/json" &&
			!(strings.HasPrefix(mediaType, "application/") && strings.HasSuffix(mediaType, "+json"))) {
			writeError(w, invalid("Expected a JSON request body"))
			return
		}
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<16))
	if err != nil {
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			writeError(w, failure{413, "Request body is too large"})
		} else {
			writeError(w, invalid("Invalid orbit or drag parameters"))
		}
		return
	}
	input := LeoDragRequest{Horizon: defaultDragHorizon}
	var fields map[string]any
	if json.Unmarshal(raw, &fields) != nil || !validBody(fields) || json.Unmarshal(raw, &input) != nil ||
		(input.Thresholds != nil && input.Thresholds.High <= input.Thresholds.Elevated) {
		writeError(w, invalid("Invalid orbit or drag parameters"))
		return
	}
	if value, present := fields[dragHorizonField]; present && value == nil {
		writeError(w, invalid("Invalid orbit or drag parameters"))
		return
	}
	body, err := json.Marshal(input)
	if err != nil {
		writeError(w, invalid("Invalid orbit or drag parameters"))
		return
	}
	raw, status, err := a.request(r.Context(), a.intelligence, "POST", "/internal/v1/public-risks/leo-drag", nil, body, 90*time.Second, dragResponseLimit)
	if status == 422 {
		writeError(w, invalid("Orbit is outside the available grid or the fixed-orbit approximation (maximum decay 1000 m)"))
		return
	}
	respondPublication(w, respond, raw, err, "Impact service or atmospheric density is not ready",
		func(assessment LeoDragAssessment) bool {
			return sameDragInput(assessment.Inputs, input) && validDragAssessment(assessment)
		})
}

func sameDragInput(result, request LeoDragRequest) bool {
	if result.Horizon != request.Horizon {
		return false
	}
	for _, pair := range [][2]*float64{
		{result.Altitude, request.Altitude}, {result.Inclination, request.Inclination},
		{result.Mass, request.Mass}, {result.Area, request.Area}, {result.Coefficient, request.Coefficient},
	} {
		if pair[0] == nil || pair[1] == nil || *pair[0] != *pair[1] {
			return false
		}
	}
	if request.Thresholds == nil {
		return result.Thresholds == nil
	}
	return result.Thresholds != nil && *result.Thresholds == *request.Thresholds
}

func validDragAssessment(assessment LeoDragAssessment) bool {
	start := assessment.StartTime
	horizon := assessment.Inputs.Horizon
	if start.IsZero() || !start.Equal(assessment.Source.IssueTime) || start.After(assessment.ComputedAt) ||
		!assessment.EndTime.Equal(start.Add(time.Duration(horizon)*time.Hour)) ||
		len(assessment.Predictions) != horizon+1 || assessment.MeanDensityKgM3 <= 0 {
		return false
	}
	for lead, point := range assessment.Predictions {
		if point.LeadHours != lead || !point.ValidTime.Equal(start.Add(time.Duration(lead)*time.Hour)) ||
			point.MeanDensityKgM3 <= 0 {
			return false
		}
	}
	last := assessment.Predictions[horizon]
	return last.DeltaVLossMS == assessment.DeltaVLossMS && last.EstimatedAltitudeLossM == assessment.EstimatedAltitudeLossM
}
