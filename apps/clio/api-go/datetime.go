package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type apiTime struct{ time.Time }

func (t apiTime) MarshalJSON() ([]byte, error) {
	return json.Marshal(formatTime(t.Time, true))
}

func (t *apiTime) UnmarshalJSON(data []byte) error {
	input := string(data)
	if bytes.HasPrefix(data, []byte{'"'}) {
		if err := json.Unmarshal(data, &input); err != nil {
			return err
		}
	}
	value, problem := parseDateTime(input)
	if problem != "" {
		return fmt.Errorf("invalid datetime: %s", problem)
	}
	t.Time = value
	return nil
}

func formatTime(t time.Time, utcZ bool) string {
	layout := "2006-01-02T15:04:05"
	if t.Nanosecond() != 0 {
		layout += ".000000"
	}
	if utcZ {
		return t.Format(layout + "Z07:00")
	}
	return t.Format(layout + "-07:00")
}

var epochPattern = regexp.MustCompile(`^[+-]?[0-9]+(?:\.[0-9]+)?$`)

const missingTimezone = "Input should have timezone info"

func parseDateTime(input string) (time.Time, string) {
	if epochPattern.MatchString(input) {
		value, err := strconv.ParseFloat(input, 64)
		if err == nil {
			if math.Abs(value) > 20_000_000_000 {
				value /= 1000
			}
			if value >= -62_135_596_800 && value < 253_402_300_800 {
				seconds, fraction := math.Modf(value)
				return time.Unix(int64(seconds), int64(math.Round(fraction*1e6))*1000).UTC(), ""
			}
		}
	}
	if len(input) < len("2006-01-02") {
		return time.Time{}, "input is too short"
	}
	date, err := time.Parse("2006-01-02", input[:10])
	if err != nil {
		for _, part := range []string{"month", "day"} {
			if strings.HasSuffix(err.Error(), part+" out of range") {
				return time.Time{}, part + " value is outside expected range"
			}
		}
		for _, part := range []struct {
			start, end int
			name       string
		}{
			{0, 4, "year"}, {5, 7, "month"}, {8, 10, "day"},
		} {
			if _, err := strconv.Atoi(input[part.start:part.end]); err != nil {
				return time.Time{}, "invalid character in " + part.name
			}
		}
		return time.Time{}, "invalid date separator, expected `-`"
	}
	if date.Year() == 0 {
		return time.Time{}, "year 0 is out of range"
	}
	if len(input) == 10 {
		return time.Time{}, missingTimezone
	}
	if strings.ContainsRune("Tt _", rune(input[10])) {
		input = input[:10] + "T" + strings.ReplaceAll(input[11:], "z", "Z")
		for _, clock := range []string{"15:04:05", "15:04"} {
			layout := "2006-01-02T" + clock
			for _, offset := range []string{"Z07:00", "Z0700"} {
				if stamp, err := time.Parse(layout+offset, input); err == nil {
					// Go accepts a 24-hour offset, which datetime does not.
					if _, offsetSeconds := stamp.Zone(); offsetSeconds > -86400 && offsetSeconds < 86400 {
						return stamp.Truncate(time.Microsecond), ""
					}
				}
			}
			if _, err := time.Parse(layout, input); err == nil {
				return time.Time{}, missingTimezone
			}
		}
	}
	return time.Time{}, "unexpected extra characters at the end of the input"
}

type queryIssue struct {
	Type    string            `json:"type"`
	Loc     [2]string         `json:"loc"`
	Message string            `json:"msg"`
	Input   *string           `json:"input"`
	Context map[string]string `json:"ctx,omitempty"`
}

type queryValidation struct {
	values url.Values
	issues []queryIssue
}

func (q *queryValidation) required(name string) (string, bool) {
	values := q.values[name]
	if len(values) != 0 {
		return values[len(values)-1], true
	}
	q.issues = append(q.issues, queryIssue{Type: "missing", Loc: [2]string{"query", name}, Message: "Field required"})
	return "", false
}

func (q *queryValidation) dateTime(name string) time.Time {
	input, ok := q.required(name)
	if !ok {
		return time.Time{}
	}
	stamp, problem := parseDateTime(input)
	if problem == "" {
		return stamp.UTC()
	}
	issue := queryIssue{Loc: [2]string{"query", name}, Input: &input}
	if problem == missingTimezone {
		issue.Type = "timezone_aware"
		issue.Message = problem
	} else {
		issue.Type = "datetime_from_date_parsing"
		issue.Message = "Input should be a valid datetime or date, " + problem
		issue.Context = map[string]string{"error": problem}
	}
	q.issues = append(q.issues, issue)
	return time.Time{}
}

func (q *queryValidation) valid(w http.ResponseWriter) bool {
	if len(q.issues) == 0 {
		return true
	}
	writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": q.issues})
	return false
}

func validRange(start, end time.Time) bool {
	delta := end.Sub(start)
	return delta > 0 && delta <= 366*24*time.Hour
}
