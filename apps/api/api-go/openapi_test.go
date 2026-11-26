package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestGeneratedOpenAPIContract(t *testing.T) {
	baseline, err := os.ReadFile("testdata/openapi-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	(&api{}).routes().ServeHTTP(response, httptest.NewRequest(http.MethodGet, prefix+"/openapi.json", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("OpenAPI returned %d: %s", response.Code, response.Body)
	}
	want, got := decodeContract(t, baseline), decodeContract(t, response.Body.Bytes())
	if !reflect.DeepEqual(got["servers"], want["servers"]) {
		t.Errorf("public base paths changed: %v", got["servers"])
	}
	wantPaths, gotPaths := contractObject(want["paths"]), contractObject(got["paths"])
	gotSchemas := contractObject(contractObject(got["components"])["schemas"])
	variable := contractObject(gotSchemas["VariableForecast"])
	update := contractObject(contractObject(variable["properties"])["update_issued_at"])
	if update["format"] != "date-time" || slices.Contains(contractStrings(variable["required"]), "update_issued_at") {
		t.Errorf("unexpected variable update timestamp schema: %v", update)
	}
	history := contractObject(contractObject(gotPaths["/observations/history"])["get"])
	timeBasis := contractObject(contractObject(contractParameters(history["parameters"])["query time_basis"])["schema"])
	if timeBasis["default"] != "l1" || !slices.Equal(contractStrings(timeBasis["enum"]), []string{"l1", "near_earth"}) {
		t.Errorf("unexpected wind time basis schema: %v", timeBasis)
	}
	current := contractObject(contractObject(gotSchemas["CurrentObservations"])["properties"])
	for _, metric := range []string{"xrs_a", "xrs_b", "proton_10mev", "proton_100mev"} {
		if !acceptsNull(current[metric]) {
			t.Errorf("current observations must expose nullable %s", metric)
		}
	}
	for _, methods := range gotPaths {
		for _, operation := range contractObject(methods) {
			if contractParameters(contractObject(operation)["parameters"])["query meta"] != nil {
				t.Errorf("unexpected metadata switch in %v", contractObject(operation)["operationId"])
			}
			for status, reply := range contractObject(contractObject(operation)["responses"]) {
				if len(status) == 3 && status[0] == '2' {
					assertNoExamples(t, reply, gotSchemas, make(map[string]bool))
				}
			}
		}
	}
	for path, methods := range wantPaths {
		for method, operation := range contractObject(methods) {
			t.Run(method+" "+path, func(t *testing.T) {
				old := contractObject(operation)
				next := contractObject(contractObject(gotPaths[path])[method])
				if old["operationId"] != next["operationId"] {
					t.Errorf("operation ID %v, want %v", next["operationId"], old["operationId"])
				}
				for _, key := range []string{"responses", "requestBody"} {
					if key == "requestBody" {
						if (old[key] == nil) != (next[key] == nil) {
							t.Errorf("request body changed")
						}
					} else {
						assertIncludes(t, "status codes", contractKeys(next[key]), contractKeys(old[key]))
					}
				}
				oldParams, nextParams := contractParameters(old["parameters"]), contractParameters(next["parameters"])
				assertIncludes(t, "parameters", contractKeys(nextParams), contractKeys(oldParams))
				for name, param := range oldParams {
					oldParam, nextParam := contractObject(param), contractObject(nextParams[name])
					oldSchema, nextSchema := contractObject(oldParam["schema"]), contractObject(nextParam["schema"])
					for _, key := range []string{"default", "minimum", "maximum"} {
						if !reflect.DeepEqual(oldSchema[key], nextSchema[key]) {
							t.Errorf("parameter %s %s = %v, want %v", name, key, nextSchema[key], oldSchema[key])
						}
					}
				}
			})
		}
	}

	wantSchemas := contractObject(contractObject(want["components"])["schemas"])
	for name, original := range wantSchemas {
		// The success envelope is generated inline around the actual response type.
		if strings.HasPrefix(name, "ApiResponse_") {
			continue
		}
		t.Run(name, func(t *testing.T) {
			old, next := contractObject(original), contractObject(gotSchemas[name])
			if next == nil {
				t.Fatal("component missing")
			}
			assertIncludes(t, "fields", contractKeys(next["properties"]), contractKeys(old["properties"]))
			assertIncludes(t, "required fields", contractStrings(next["required"]), contractStrings(old["required"]))
			for field, shape := range contractObject(old["properties"]) {
				newShape := contractObject(next["properties"])[field]
				if acceptsNull(shape) != acceptsNull(newShape) {
					t.Errorf("field %s changed its nullability: %v", field, newShape)
				}
			}
		})
	}
}

func decodeContract(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var result map[string]any
	if err := json.Unmarshal(raw, &result); err != nil {
		t.Fatal(err)
	}
	return result
}

func assertIncludes(t *testing.T, kind string, current, baseline []string) {
	t.Helper()
	for _, name := range baseline {
		if !slices.Contains(current, name) {
			t.Errorf("%s no longer includes %s", kind, name)
		}
	}
}

func assertNoExamples(t *testing.T, value any, schemas map[string]any, visited map[string]bool) {
	t.Helper()
	switch node := value.(type) {
	case map[string]any:
		if ref, ok := node["$ref"].(string); ok && !visited[ref] {
			visited[ref] = true
			name, ok := strings.CutPrefix(ref, componentPrefix)
			if !ok || schemas[name] == nil {
				t.Errorf("unresolved response schema %q", ref)
			}
			assertNoExamples(t, schemas[name], schemas, visited)
		}
		for key, value := range node {
			if key == "example" || key == "examples" {
				t.Errorf("explicit example in a successful public response")
			}
			assertNoExamples(t, value, schemas, visited)
		}
	case []any:
		for _, value := range node {
			assertNoExamples(t, value, schemas, visited)
		}
	}
}

func contractObject(value any) map[string]any {
	result, _ := value.(map[string]any)
	return result
}

func contractKeys(value any) []string {
	var keys []string
	for key := range contractObject(value) {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func contractStrings(value any) []string {
	items, _ := value.([]any)
	var result []string
	for _, item := range items {
		result = append(result, item.(string))
	}
	slices.Sort(result)
	return result
}

func contractParameters(value any) map[string]any {
	items, _ := value.([]any)
	result := make(map[string]any, len(items))
	for _, item := range items {
		param := contractObject(item)
		result[param["in"].(string)+" "+param["name"].(string)] = param
	}
	return result
}

func acceptsNull(value any) bool {
	schema := contractObject(value)
	if schema == nil {
		return false
	}
	if schema["type"] == "null" || slices.Contains(contractStrings(schema["type"]), "null") {
		return true
	}
	options, _ := schema["anyOf"].([]any)
	for _, option := range options {
		if acceptsNull(option) {
			return true
		}
	}
	return false
}
