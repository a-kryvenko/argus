package publicforecast

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
)

type Row map[string]string

type Variable struct {
	Name, Artifact, Unit, Column string
	Quantiles                    bool
	Thresholds                   []int
}

type Product struct {
	Horizon   int
	Variables []Variable
}

var Products = map[string]Product{
	"solar-wind-speed": {96, []Variable{
		{"v", "plasma_speed_quantile", "km/s", "v", true, nil},
		{"v", "plasma_speed_threshold", "km/s", "v", false, []int{450, 500, 600}},
	}},
	"solar-wind-density":     {96, []Variable{{"n", "plasma_density_quantile", "cm^-3", "n", true, nil}}},
	"solar-wind-temperature": {96, []Variable{{"t", "plasma_temperature_quantile", "K", "t", true, nil}}},
	"hmf": {48, []Variable{
		{"bt", "hmf_total_threshold", "nT", "bt", false, []int{5, 10, 15}},
		{"bs", "hmf_southward_threshold", "nT", "bs", false, []int{5, 10, 15}},
	}},
	"solar-radiation": {48, []Variable{
		{"f10_7", "f10_7_quantile", "sfu", "f107", true, nil},
		{"s10", "s10_quantile", "index", "s10", true, nil},
		{"m10", "m10_quantile", "index", "m10", true, nil},
		{"y10", "y10_quantile", "index", "y10", true, nil},
	}},
	"geomagnetic-activity": {48, []Variable{
		{"kp", "kp_threshold", "index", "kp", false, []int{4, 5, 6}},
		{"ap", "ap_quantile", "index", "ap", true, nil},
	}},
	"dst": {48, []Variable{{"dst", "dst_quantile", "nT", "dst", true, nil}}},
}

func ParseTimestamp(value string) (time.Time, error) {
	// Historical CSVs may separate the date and time with a space.
	if len(value) > 10 && value[10] == ' ' {
		value = value[:10] + "T" + value[11:]
	}
	at, err := time.Parse(time.RFC3339Nano, value)
	return at.UTC(), err
}

func number(value string) (float64, error) {
	result, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsNaN(result) || math.IsInf(result, 0) {
		return 0, errors.New("invalid numeric value")
	}
	return result, nil
}

func ForecastData(target string, product Product, frames map[string][]Row, issue time.Time) (*Forecast, error) {
	points := make(map[int]*ForecastPoint)
	available := []string{}
	for _, variable := range product.Variables {
		if !slices.Contains(available, variable.Name) {
			available = append(available, variable.Name)
		}
		for _, row := range frames[variable.Artifact] {
			lead, err := strconv.Atoi(row["lead_hours"])
			valid, timeErr := ParseTimestamp(row["valid_time"])
			if err != nil || timeErr != nil || lead < 0 || !valid.Equal(issue.Add(time.Duration(lead)*time.Hour)) {
				return nil, errors.New("inconsistent forecast lead time")
			}
			if lead > product.Horizon {
				continue
			}
			if points[lead] == nil {
				points[lead] = &ForecastPoint{Valid: valid, Lead: lead, Variables: make(map[string]*VariableForecast)}
			}
			value := points[lead].Variables[variable.Name]
			if value == nil {
				value = &VariableForecast{Binary: []BinaryForecast{}}
				points[lead].Variables[variable.Name] = value
			}
			if err := addPrediction(value, variable, row); err != nil {
				return nil, err
			}
		}
	}
	if len(points) == 0 {
		return nil, errors.New("empty forecast")
	}
	predictions := make([]*ForecastPoint, 0, len(points))
	times := make(map[time.Time]bool, len(points))
	for lead := 0; lead <= product.Horizon; lead++ {
		if point := points[lead]; point != nil {
			predictions = append(predictions, point)
			times[point.Valid] = true
		}
	}
	data := &Forecast{Target: target, IssueTime: issue, Predictions: predictions,
		HorizonHours: predictions[len(predictions)-1].Lead, AvailableVariables: available}
	data.Meta = Metadata(product, available)
	if target == "solar-wind-speed" {
		outlook, err := cmeOutlook(frames["plasma_speed_quantile"], times)
		if err != nil {
			return nil, err
		}
		data.CMESpeed = outlook
	}
	return data, nil
}

func addPrediction(value *VariableForecast, variable Variable, row Row) error {
	if variable.Quantiles {
		value.Continuous = &QuantileForecast{}
		for _, quantile := range []struct {
			column string
			value  *float64
		}{{"q10", &value.Continuous.Q10}, {"q50", &value.Continuous.Q50}, {"q90", &value.Continuous.Q90}} {
			v, err := number(row[variable.Column+"_"+quantile.column])
			if err != nil {
				return err
			}
			switch variable.Name {
			case "v":
				v = math.RoundToEven(v)
			case "n":
				v, _ = strconv.ParseFloat(strconv.FormatFloat(v, 'f', 3, 64), 64)
			}
			*quantile.value = v
		}
	}
	for _, threshold := range variable.Thresholds {
		encoded, found := row[fmt.Sprintf("p_%s_ge_%d", variable.Column, threshold)]
		if !found {
			return errors.New("missing probability column")
		}
		if encoded == "" || strings.EqualFold(encoded, "nan") {
			continue
		}
		probability, err := number(encoded)
		if err != nil || probability < 0 || probability > 1 {
			return errors.New("invalid probability")
		}
		value.Binary = append(value.Binary, BinaryForecast{float64(threshold), probability})
	}
	return nil
}

func cmeOutlook(rows []Row, times map[time.Time]bool) (*Published[CMESpeedOutlook], error) {
	var encoded string
	for _, row := range rows {
		value := row["cme_speed_json"]
		if value == "" || strings.EqualFold(value, "nan") {
			continue
		}
		if encoded != "" {
			return nil, errors.New("multiple CME outlooks in a single forecast")
		}
		encoded = value
	}
	if encoded == "" {
		return nil, nil
	}
	var outlook map[string]json.RawMessage
	var points []json.RawMessage
	if json.Unmarshal([]byte(encoded), &outlook) != nil || json.Unmarshal(outlook["points"], &points) != nil || points == nil {
		return nil, errors.New("invalid CME outlook")
	}
	selected := make([]json.RawMessage, 0, len(points))
	for _, raw := range points {
		var point CMESpeedPoint
		if json.Unmarshal(raw, &point) != nil || point.ValidTime.IsZero() {
			return nil, errors.New("invalid CME outlook time")
		}
		if times[point.ValidTime.UTC()] {
			selected = append(selected, raw)
		}
	}
	outlook["points"], _ = json.Marshal(selected)
	raw, err := json.Marshal(outlook)
	if err != nil {
		return nil, err
	}
	return &Published[CMESpeedOutlook]{raw}, nil
}

func Metadata(product Product, available []string) *ForecastMetadata {
	variables := make(map[string]ForecastVariableMetadata)
	for _, variable := range product.Variables {
		if slices.Contains(available, variable.Name) {
			variables[variable.Name] = ForecastVariableMetadata{Unit: variable.Unit}
		}
	}
	return &ForecastMetadata{Variables: variables}
}
