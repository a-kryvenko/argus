package publicforecast

import (
	"encoding/json"
	"errors"
	"strconv"
	"time"
)

type densityCoordinate struct{ Altitude, Latitude float64 }

func DensityData(rows []Row, issue time.Time) (*DensityForecast, error) {
	if len(rows) == 0 {
		return nil, errors.New("empty density forecast")
	}
	first := rows[0]
	model, hasModel := first["model"]
	if !hasModel {
		model = "JB2008"
	}
	magneticDate, magneticMethod := "dtc_observed_at", "dtc_method"
	method := "causal_dst_ap_v1"
	switch model {
	case "DTM2020":
		magneticDate, magneticMethod = "geomagnetic_observed_at", "geomagnetic_method"
		method = "ap_to_kp_lag3h_mean24h_v1"
	case "JB2008":
	default:
		return nil, errors.New("unsupported density model")
	}
	if first["driver_mode"] != "observed_persistence" || first[magneticMethod] != method {
		return nil, errors.New("invalid density driver metadata")
	}
	observed, observedErr := ParseTimestamp(first["observed_at"])
	history, historyErr := ParseTimestamp(first["history_start"])
	magnetic, magneticErr := ParseTimestamp(first[magneticDate])
	if observedErr != nil || historyErr != nil || magneticErr != nil || observed.After(issue) ||
		history.After(observed) || magnetic.After(issue) {
		return nil, errors.New("invalid density timestamps")
	}
	interpolated, meanDates, err := densityBackground(first, model, issue)
	if err != nil {
		return nil, err
	}
	reference, err := densityReference(rows, model)
	if err != nil {
		return nil, err
	}
	points := make([]DensityPoint, 49)
	grid := make([]map[densityCoordinate]bool, len(points))
	latitudes, altitudes := make(map[float64]bool), make(map[float64]bool)
	for _, row := range rows {
		for _, name := range []string{"model", "driver_mode", "background_method", "background_interpolated_days",
			"background_mean_observed_at", "history_start", "observed_at", magneticDate, magneticMethod} {
			if row[name] != first[name] {
				return nil, errors.New("inconsistent density metadata")
			}
		}
		if model == "DTM2020" && (row["dtc_method"] != "" || row["dtc_observed_at"] != "") {
			return nil, errors.New("DTM artifact contains DTC metadata")
		}
		lead, err := strconv.Atoi(row["lead_hours"])
		at, atErr := ParseTimestamp(row["valid_time"])
		if err != nil || atErr != nil || lead < 0 || lead >= len(points) || !at.Equal(issue.Add(time.Duration(lead)*time.Hour)) {
			return nil, errors.New("invalid density lead time")
		}
		cell, coordinate, err := densityCell(row, reference != nil)
		if err != nil {
			return nil, err
		}
		if grid[lead] == nil {
			grid[lead] = make(map[densityCoordinate]bool)
			points[lead] = DensityPoint{Valid: at, Lead: lead}
		}
		if grid[lead][coordinate] {
			return nil, errors.New("duplicate density cell")
		}
		grid[lead][coordinate] = true
		latitudes[coordinate.Latitude], altitudes[coordinate.Altitude] = true, true
		points[lead].Cells = append(points[lead].Cells, cell)
	}
	for _, coordinates := range grid {
		if len(coordinates) != len(latitudes)*len(altitudes) {
			return nil, errors.New("incomplete density grid")
		}
	}
	data := &DensityForecast{Target: "atmospheric-density", DriverMode: "observed_persistence",
		IssueTime: issue, ObservedAt: observed, GeomagneticObservedAt: &magnetic,
		BackgroundInterpolated: len(interpolated) > 0, HorizonHours: 48, Predictions: points, Reference: reference}
	if model == "JB2008" {
		data.DTCObservedAt = &magnetic
	}
	data.Meta = &DensityMetadata{Model: model, HistoryStart: history, BackgroundMethod: first["background_method"],
		BackgroundInterpolatedDays: interpolated, BackgroundMeanObservedAt: meanDates, GeomagneticMethod: &method}
	if model == "JB2008" {
		data.Meta.DTCMethod = &method
	}
	return data, nil
}

func densityBackground(row Row, model string, issue time.Time) (map[string][]CalendarDate, map[string]time.Time, error) {
	days, means := make(map[string][]CalendarDate), make(map[string]time.Time)
	for name, target := range map[string]any{"background_interpolated_days": &days, "background_mean_observed_at": &means} {
		if encoded := row[name]; encoded != "" && json.Unmarshal([]byte(encoded), target) != nil {
			return nil, nil, errors.New("invalid density background metadata")
		}
	}
	if days == nil || means == nil {
		return nil, nil, errors.New("missing density background metadata")
	}
	drivers := map[string]bool{"f10_7": true, "s10": model == "JB2008", "m10": model == "JB2008", "y10": model == "JB2008"}
	switch row["background_method"] {
	case "trailing_81_daily_values":
		if len(days) == 0 && len(means) == 0 {
			return days, means, nil
		}
	case "trailing_81_daily_values_linear_gapfill":
		if len(days) > 0 && len(means) == 0 {
			for driver, dates := range days {
				if !drivers[driver] || len(dates) < 1 || len(dates) > 2 {
					return nil, nil, errors.New("invalid density interpolation")
				}
				for _, date := range dates {
					if _, err := time.Parse(time.DateOnly, string(date)); err != nil {
						return nil, nil, err
					}
				}
			}
			return days, means, nil
		}
	case "published_81_day_mean_persistence":
		if model == "JB2008" && len(days) == 0 && len(means) == len(drivers) {
			for driver, at := range means {
				if !drivers[driver] || at.IsZero() || at.After(issue) {
					return nil, nil, errors.New("invalid published density background")
				}
			}
			return days, means, nil
		}
	}
	return nil, nil, errors.New("inconsistent density background method")
}

func densityReference(rows []Row, model string) (*DensityBackground, error) {
	first := rows[0]
	_, method := first["reference_method"]
	_, ap := first["reference_ap"]
	_, dtc := first["reference_dtc_kelvin"]
	if !method && !ap && !dtc {
		for _, row := range rows {
			if row["rho_reference_kg_m3"] != "" {
				return nil, errors.New("incomplete density reference")
			}
		}
		return nil, nil
	}
	geomagneticAP, apErr := number(first["reference_ap"])
	kelvin, kelvinErr := number(first["reference_dtc_kelvin"])
	if model != "JB2008" || first["reference_method"] != "jb2008_fixed_ap4_v1" ||
		apErr != nil || geomagneticAP != 4 || kelvinErr != nil || kelvin < 0 {
		return nil, errors.New("invalid density reference")
	}
	for _, row := range rows {
		for _, key := range []string{"reference_method", "reference_ap", "reference_dtc_kelvin"} {
			if first[key] != row[key] {
				return nil, errors.New("inconsistent density reference")
			}
		}
	}
	return &DensityBackground{Method: first["reference_method"], GeomagneticAP: geomagneticAP,
		DTCKelvin: kelvin, SpatialSupport: "longitude_mean", SolarDrivers: "same_as_forecast"}, nil
}

func densityCell(row Row, hasReference bool) (DensityCell, densityCoordinate, error) {
	altitude, altitudeErr := number(row["altitude_km"])
	latitude, latitudeErr := number(row["latitude_deg"])
	if altitudeErr != nil || latitudeErr != nil || altitude < 90 || altitude > 2500 || latitude < -90 || latitude > 90 {
		return DensityCell{}, densityCoordinate{}, errors.New("invalid density grid coordinate")
	}
	cell := DensityCell{AltitudeKm: altitude, LatitudeDeg: latitude}
	fields := []struct {
		key   string
		value *float64
	}{
		{"rho_kg_m3", &cell.RhoKgM3}, {"rho_lon_p10_kg_m3", &cell.RhoLonP10KgM3},
		{"rho_lon_p90_kg_m3", &cell.RhoLonP90KgM3},
	}
	if hasReference {
		cell.RhoReferenceKgM3 = new(float64)
		fields = append(fields, struct {
			key   string
			value *float64
		}{"rho_reference_kg_m3", cell.RhoReferenceKgM3})
	}
	for _, field := range fields {
		value, err := number(row[field.key])
		if err != nil || value <= 0 {
			return DensityCell{}, densityCoordinate{}, errors.New("invalid density value")
		}
		*field.value = value
	}
	return cell, densityCoordinate{altitude, latitude}, nil
}
