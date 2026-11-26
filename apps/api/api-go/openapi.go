package main

import (
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
	"github.com/a-kryvenko/argus/observationcatalog"
	"github.com/danielgtaylor/huma/v2"
)

const componentPrefix = "#/components/schemas/"

func init() {
	huma.DefaultArrayNullable = false
}

func newOpenAPI() *huma.OpenAPI {
	return &huma.OpenAPI{
		OpenAPI: "3.1.0",
		Info: &huma.Info{
			Title: "ARGUS SUNWATCH API", Version: "0.1.0",
			Description: "Public observations, forecasts and orbital risk assessments. These operations are available without a dashboard account.",
		},
		Servers: []*huma.Server{{URL: prefix}},
		Components: &huma.Components{
			Schemas: huma.NewMapRegistry(componentPrefix, huma.DefaultSchemaNamer),
		},
	}
}

func completeSchemas(registry huma.Registry) {
	null := &huma.Schema{Type: "null", Enum: []any{nil}}
	null.PrecomputeMessages()
	for name, schema := range registry.Map() {
		typ := registry.TypeFromRef(componentPrefix + name)
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			key, _, _ := strings.Cut(field.Tag.Get("json"), ",")
			property := schema.Properties[key]
			if field.Tag.Get("allowNull") == "true" {
				// Keep the referenced object shared by its nullable and required uses.
				schema.Properties[key] = &huma.Schema{Description: property.Description,
					AnyOf: []*huma.Schema{property, null}}
			}
			if property != nil && property.Nullable && len(property.Enum) > 0 {
				property.Enum = append(property.Enum, nil)
			}
		}
		if name != "LeoDragRequest" && name != "DragThresholds" && name != "DragPoint" {
			schema.AdditionalProperties = nil
		}
	}
	registry.SchemaFromRef(componentPrefix + "PublicError").Properties["data"] = &huma.Schema{Type: "null"}
}

const (
	trendsParameter         = "trends"
	observationsParameter   = "observations"
	limitParameter          = "limit"
	fromParameter           = "from"
	toParameter             = "to"
	resolutionParameter     = "resolution"
	timeBasisParameter      = "time_basis"
	targetParameter         = "target"
	defaultObservationLimit = 24
	maxObservationLimit     = 168
	defaultResolution       = "1m"
	dragHorizonField        = "horizon_hours"
	defaultDragHorizon      = 24
	pingReply               = "pong"
)

var observationResolutions = []string{"1m", "5m", "1h", "auto"}

func (a *api) publicRoutes(mux *http.ServeMux, spec *huma.OpenAPI) {
	trends := queryParameter(trendsParameter, "Include observed one-hour changes", &huma.Schema{Type: "boolean", Default: false})
	observations := queryParameter(observationsParameter, "Comma-separated observations: "+strings.Join(allMetrics, ","), &huma.Schema{Type: "string"})
	minLimit, maxLimit := float64(1), float64(maxObservationLimit)
	historyParameters := []*huma.Param{
		queryParameter(limitParameter, "Maximum number of rows in the background index points array.", &huma.Schema{
			Type: "integer", Default: defaultObservationLimit, Minimum: &minLimit, Maximum: &maxLimit,
		}),
		queryParameter(fromParameter, "Start time with timezone.", &huma.Schema{Type: "string", Format: "date-time"}),
		queryParameter(toParameter, "End time with timezone.", &huma.Schema{Type: "string", Format: "date-time"}),
		queryParameter(resolutionParameter, "Solar-wind resolution. For L1, auto selects 1m through 6 hours, 5m through 24 hours and 1h for longer ranges. Near-Earth wind defaults to 1h.",
			&huma.Schema{Type: "string", Default: defaultResolution, Enum: enumValues(observationResolutions)}),
		queryParameter(timeBasisParameter, "Solar-wind time basis. l1 uses NOAA spacecraft measurement time. near_earth uses stored hourly NASA OMNI2 observations on the propagated near-Earth time axis and may have gaps.",
			&huma.Schema{Type: "string", Default: l1TimeBasis, Enum: []any{l1TimeBasis, nearEarthTimeBasis}}),
		observations,
	}
	targets := slices.Sorted(maps.Keys(publicforecast.Products))
	target := &huma.Param{Name: targetParameter, In: "path", Required: true,
		Description: "Forecast product identifier.", Schema: &huma.Schema{Type: "string", Enum: enumValues(targets)}}

	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/ping", OperationID: "ping", Tags: []string{"ping"},
		Summary: "Ping"}, a.ping)
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/observations", OperationID: "currentObservations",
		Tags: []string{"observations"}, Summary: "Current observations", Parameters: []*huma.Param{trends, observations},
		Description: "Latest available observations, including NOAA L1 solar wind and NOAA primary GOES X-ray and integral proton fluxes.",
		Errors:      []int{422, 503}}, a.currentObservations)
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/observations/history", OperationID: "observationHistory",
		Tags: []string{"observations"}, Summary: "Observation history", Parameters: historyParameters, Errors: []int{422, 503},
		Description: "Missing measurements remain gaps. Times must include a timezone. Wind query ranges are limited to 7 days at 1m, 31 days at 5m and 366 days at 1h. Kp and Dst history is limited to 31 days. GOES history requires an explicit observations selection and accepts ranges of up to 7 days. XRS uses 1-minute samples in W/m². Proton flux uses 5-minute samples in pfu. The limit parameter applies to the background index points array."}, a.observationHistory)
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/observations/catalog", OperationID: "observationCatalog",
		Tags: []string{"observations"}, Summary: "Observation catalog",
		Description: "Metric identifiers, units, native cadence and live products."},
		func(w http.ResponseWriter, _ *http.Request, respond func(observationcatalog.Catalog)) {
			w.Header().Set("Cache-Control", "public, max-age=3600")
			metrics := make([]observationcatalog.Metric, 0, len(allMetrics))
			for _, metric := range observationcatalog.Metrics {
				if slices.Contains(allMetrics, metric.ID) {
					metrics = append(metrics, metric)
				}
			}
			respond(observationcatalog.Catalog{Metrics: metrics, ProductURLs: observationcatalog.ProductURLs})
		})
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/summary", OperationID: "summary",
		Tags: []string{"summary"}, Summary: "Get the latest published summary", Errors: []int{503},
		Description: "Observation groups, chart links and published CME arrival windows."}, a.summary)
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/forecasts/atmospheric-density", OperationID: "atmosphericDensity",
		Tags: []string{"forecasts"}, Summary: "Get the latest atmospheric density forecast",
		Errors: []int{503}, Description: "Returns a precomputed 48-hour thermospheric density grid. Drivers are held constant at issue time. Backgrounds use trailing 81-day means. Longitude percentiles describe spatial variation. Returns 503 when the grid is missing, invalid or stale."}, a.density)
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/forecasts/solar-wind-events", OperationID: "solarWindEvents",
		Tags: []string{"forecasts"}, Summary: "Get CME arrivals and conditional solar-wind speed forecasts", Errors: []int{503},
		Description: "Published DONKI/ENLIL Earth impacts with a separate CME-conditioned speed outlook. Nominal, early and late scenarios retain the event peak. Scenario envelopes are not calibrated quantiles or hit probabilities. HSS forecasts are disabled in the current release."}, a.solarWindEvents)
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/forecasts/{target}", OperationID: "forecast",
		Tags: []string{"forecasts"}, Summary: "Get a forecast", Parameters: []*huma.Param{target}, Errors: []int{404, 503},
		Description: forecastDescription(targets)}, a.forecast)
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/forecasts/{target}/metrics", OperationID: "forecastMetrics",
		Tags: []string{"forecasts"}, Summary: "Get forecast metrics", Parameters: []*huma.Param{target}, Errors: []int{404, 503},
		Description: forecastDescription(targets)}, a.metrics)
	publicRoute(mux, spec, huma.Operation{Method: http.MethodGet, Path: "/forecasts/{target}/verification", OperationID: "forecastVerification",
		Tags: []string{"forecasts"}, Summary: "Get last 30 days published forecast accuracy", Parameters: []*huma.Param{target},
		Errors: []int{404, 503}}, a.verification)
	registry := spec.Components.Schemas
	requestSchema := schemaFor[LeoDragRequest](registry)
	registry.SchemaFromRef(requestSchema.Ref).Properties[dragHorizonField].Default = defaultDragHorizon
	publicRoute(mux, spec, huma.Operation{Method: http.MethodPost, Path: "/risks/leo-drag", OperationID: "assessLeoDrag",
		Tags: []string{"risks"}, Summary: "Estimate drag for a circular LEO orbit",
		Description: "First-order drag assessment for 200–800 km using a fresh density grid. The 24/48-hour horizon starts at forecast issue time. Optional cumulative altitude-loss thresholds determine risk categories. Returns 422 outside the grid or above 1000 m estimated decay.",
		RequestBody: &huma.RequestBody{Required: true, Content: schemaContent(requestSchema)},
		Errors:      []int{413, 422, 503}}, func(w http.ResponseWriter, r *http.Request, respond func(Published[LeoDragAssessment])) {
		a.leoDrag(w, r, respond, func(body any) bool {
			var result huma.ValidateResult
			huma.Validate(registry, requestSchema, huma.NewPathBuffer(nil, 0), huma.ModeWriteToServer, body, &result)
			return len(result.Errors) == 0
		})
	})
}

func enumValues(values []string) []any {
	result := make([]any, len(values))
	for i, value := range values {
		result[i] = value
	}
	return result
}

func queryParameter(name, description string, schema *huma.Schema) *huma.Param {
	return &huma.Param{Name: name, In: "query", Description: description, Schema: schema}
}

func schemaFor[T any](registry huma.Registry) *huma.Schema {
	return registry.Schema(reflect.TypeFor[T](), true, "")
}

func schemaContent(schema *huma.Schema) map[string]*huma.MediaType {
	return map[string]*huma.MediaType{"application/json": {Schema: schema}}
}

func publicRoute[T any](mux *http.ServeMux, spec *huma.OpenAPI, op huma.Operation, handler func(http.ResponseWriter, *http.Request, func(T))) {
	registry := spec.Components.Schemas
	body := schemaFor[T](registry)
	if op.OperationID == "ping" {
		body.Enum = []any{pingReply}
	}
	response := &huma.Schema{Type: "object", Required: []string{"success", "data", "error"}, Properties: map[string]*huma.Schema{
		"success": {Type: "boolean", Enum: []any{true}}, "data": body, "error": {Type: "null"},
	}}
	op.Responses = map[string]*huma.Response{"200": {Description: "Successful response", Content: schemaContent(response)}}
	for _, status := range op.Errors {
		op.Responses[strconv.Itoa(status)] = &huma.Response{Description: http.StatusText(status),
			Content: schemaContent(schemaFor[PublicError](registry))}
	}
	spec.AddOperation(&op)
	registerRoute(mux, op.Method, op.Path, func(w http.ResponseWriter, r *http.Request) {
		handler(w, r, func(result T) { success(w, result) })
	})
}

// Published retains the original JSON from a producing service after its
// typed publication checks.
type Published[T any] struct{ raw json.RawMessage }

func (p Published[T]) MarshalJSON() ([]byte, error) { return p.raw.MarshalJSON() }

func (Published[T]) Schema(registry huma.Registry) *huma.Schema { return schemaFor[T](registry) }

func forecastDescription(targets []string) string {
	var text strings.Builder
	text.WriteString("Available targets: " + strings.Join(targets, ", ") + ".\n\n")
	text.WriteString("| Target | Variable (unit) | Forecast type | Maximum horizon (hours) |\n| --- | --- | --- | --- |\n")
	for _, name := range targets {
		product := publicforecast.Products[name]
		for _, variable := range product.Variables {
			kind := "Quantiles q10, q50, q90"
			if !variable.Quantiles {
				thresholds := make([]string, len(variable.Thresholds))
				for i, threshold := range variable.Thresholds {
					thresholds[i] = strconv.Itoa(threshold)
				}
				kind = "Probability of value ≥ " + strings.Join(thresholds, ", ")
			}
			fmt.Fprintf(&text, "| %s | %s (%s) | %s | %d |\n", name, variable.Name, variable.Unit, kind, product.Horizon)
		}
	}
	return text.String()
}
