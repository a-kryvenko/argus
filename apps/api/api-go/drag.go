package main

import (
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

func (a *api) leoDrag(w http.ResponseWriter, r *http.Request, respond func(Published[LeoDragAssessment]), validBody func(any) bool) {
	meta, err := booleanQuery(r.URL.Query(), metaParameter)
	if err != nil {
		writeError(w, err)
		return
	}
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
	query := url.Values{metaParameter: {strconv.FormatBool(meta)}}
	raw, status, err := a.request(r.Context(), a.intelligence, "POST", "/internal/v1/public-risks/leo-drag", query, body, 90*time.Second)
	if status == 422 {
		writeError(w, invalid("Orbit is outside the available grid or the fixed-orbit approximation (maximum decay 1000 m)"))
		return
	}
	if err != nil {
		writeError(w, unavailable("Impact service or atmospheric density is not ready"))
		return
	}
	respond(Published[LeoDragAssessment]{raw})
}
