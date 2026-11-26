package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const prefix = "/api/v1"

const (
	latestResponseLimit  = 64 << 10
	historyResponseLimit = 64 << 20
	densityResponseLimit = 32 << 20
	reportResponseLimit  = 4 << 20
	dragResponseLimit    = 256 << 10
)

type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type PublicError struct {
	Success bool     `json:"success" enum:"false"`
	Data    any      `json:"data" required:"false"`
	Error   apiError `json:"error"`
}

type failure struct {
	status  int
	message string
}

func (e failure) Error() string { return e.message }

func invalid(message string) error     { return failure{422, message} }
func unavailable(message string) error { return failure{503, message} }

func registerRoute(mux *http.ServeMux, method, path string, handler http.HandlerFunc) {
	mux.HandleFunc(prefix+path, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			w.Header().Set("Allow", method)
			writeError(w, failure{405, "Method Not Allowed"})
			return
		}
		handler(w, r)
	})
}

func (a *api) routes() http.Handler {
	mux := http.NewServeMux()
	spec := newOpenAPI()

	a.publicRoutes(mux, spec)

	completeSchemas(spec.Components.Schemas)

	openAPI, err := json.Marshal(spec)
	if err != nil {
		panic(fmt.Errorf("encode OpenAPI: %w", err))
	}

	for _, path := range []string{
		prefix + "/health/live",
		prefix + "/health/ready",
	} {
		mux.HandleFunc("GET "+path, func(w http.ResponseWriter, _ *http.Request) {
			writeJSON(w, 200, map[string]string{"service": "public-api", "status": "ok"})
		})
	}

	registerRoute(mux, "GET", "/openapi.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write(openAPI)
	})
	registerRoute(mux, "GET", "/docs", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		io.WriteString(w, swaggerHTML)
	})
	registerRoute(mux, "GET", "/redoc", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		io.WriteString(w, redocHTML)
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Match the old public API's trailing-slash redirect only for known routes.
		if path, found := strings.CutSuffix(r.URL.Path, "/"); found && path != "" {
			request := r.Clone(r.Context())
			request.URL.Path = path
			if _, pattern := mux.Handler(request); pattern != "/" && pattern != "" {
				r.URL.Path, r.URL.RawPath = path, ""
				http.Redirect(w, r, r.URL.RequestURI(), http.StatusTemporaryRedirect)
				return
			}
		}
		writeError(w, failure{404, "Not Found"})
	})

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		w.Header().Add("Vary", "Origin")
		if origin == "http://localhost:3000" || origin == "http://127.0.0.1:3000" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			if r.Method == http.MethodOptions && r.Header.Get("Access-Control-Request-Method") != "" {
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", r.Header.Get("Access-Control-Request-Headers"))
				w.Header().Set("Access-Control-Max-Age", "600")
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}
		ctx, cancel := context.WithTimeout(r.Context(), 105*time.Second)
		defer cancel()
		mux.ServeHTTP(w, r.WithContext(ctx))
	})
}

const swaggerHTML = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ARGUS SUNWATCH API</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"></head><body><div id="swagger-ui"></div><script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script><script>SwaggerUIBundle({url:"/api/v1/openapi.json",dom_id:"#swagger-ui",validatorUrl:null,defaultModelRendering:"model"});</script></body></html>`
const redocHTML = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ARGUS SUNWATCH API</title></head><body><redoc spec-url="/api/v1/openapi.json"></redoc><script src="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"></script></body></html>`

func (a *api) ping(_ http.ResponseWriter, _ *http.Request, respond func(string)) { respond(pingReply) }

func success[T any](w http.ResponseWriter, data T) {
	writeJSON(w, 200, struct {
		Success bool      `json:"success"`
		Data    T         `json:"data"`
		Error   *apiError `json:"error"`
	}{Success: true, Data: data})
}

func writeError(w http.ResponseWriter, err error) {
	var e failure
	if !errors.As(err, &e) {
		log.Printf("public API: %v", err)
		e = failure{500, "Something went wrong..."}
	}
	code := map[int]string{404: "NOT_FOUND", 503: "NOT_READY", 500: "INTERNAL_ERROR"}[e.status]
	if code == "" {
		code = "HTTP_ERROR"
	}
	writeJSON(w, e.status, PublicError{Error: apiError{code, e.message}})
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	body, err := json.Marshal(data)
	if err != nil {
		log.Printf("encode public API response: %v", err)
		status = 500
		body = []byte(`{"success":false,"data":null,"error":{"code":"INTERNAL_ERROR","message":"Something went wrong..."}}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := w.Write(body); err != nil {
		log.Printf("write public API response: %v", err)
	}
}

func booleanQuery(query url.Values, name string) (bool, error) {
	values := query[name]
	if len(values) == 0 {
		return false, nil
	}
	switch strings.ToLower(values[len(values)-1]) {
	case "1", "true", "t", "on", "yes", "y":
		return true, nil
	case "0", "false", "f", "off", "no", "n":
		return false, nil
	default:
		return false, invalid(name + " must be a boolean")
	}
}

func (a *api) request(ctx context.Context, upstream upstream, method, path string, query url.Values, body []byte, timeout time.Duration, limit int64) (json.RawMessage, int, error) {
	if upstream.url == "" || upstream.token == "" {
		return nil, 0, errors.New("upstream is not configured")
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	address := upstream.url + path
	if len(query) > 0 {
		address += "?" + query.Encode()
	}
	request, err := http.NewRequestWithContext(ctx, method, address, bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	request.Header.Set("Authorization", "Bearer "+upstream.token)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := a.client.Do(request)
	if err != nil {
		return nil, 0, err
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		return nil, response.StatusCode, fmt.Errorf("upstream returned %d", response.StatusCode)
	}
	if response.ContentLength > limit {
		return nil, response.StatusCode, errors.New("upstream response exceeds size limit")
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return nil, response.StatusCode, err
	}
	if int64(len(raw)) > limit {
		return nil, response.StatusCode, errors.New("upstream response exceeds size limit")
	}
	return raw, response.StatusCode, nil
}

func smoke(base, upstream string) error {
	client := &http.Client{Timeout: 30 * time.Second}
	read := func(path string) ([]byte, string, error) {
		response, err := client.Get(base + prefix + path)
		if err != nil {
			return nil, "", err
		}
		defer response.Body.Close()
		if response.StatusCode != 200 {
			return nil, "", fmt.Errorf("%s returned HTTP %d", path, response.StatusCode)
		}
		body, err := io.ReadAll(io.LimitReader(response.Body, 16<<20))
		return body, response.Header.Get("X-Argus-Upstream"), err
	}
	for deadline := time.Now().Add(5 * time.Second); ; {
		_, serving, err := read("/health/ready")
		if err != nil {
			return err
		}
		if upstream == "" || serving == upstream {
			break
		}
		if time.Now().After(deadline) {
			return errors.New("router did not select the expected public API slot")
		}
		time.Sleep(100 * time.Millisecond)
	}
	paths := []string{"/ping", "/openapi.json", "/docs"}
	if upstream != "" {
		// The direct slot still reaches dependencies in the active color.
		paths = append(paths, "/observations", "/forecasts/solar-wind-speed")
	}
	for _, path := range paths {
		body, serving, err := read(path)
		if err != nil {
			return err
		}
		if upstream != "" && serving != upstream {
			return fmt.Errorf("%s reached a different public API slot", path)
		}
		switch path {
		case "/openapi.json":
			var spec struct {
				Paths map[string]json.RawMessage `json:"paths"`
			}
			if json.Unmarshal(body, &spec) != nil || spec.Paths["/observations"] == nil {
				return errors.New("invalid public OpenAPI document")
			}
		case "/docs":
			if !bytes.Contains(body, []byte(prefix+"/openapi.json")) {
				return errors.New("API documentation does not reference OpenAPI")
			}
		default:
			var payload struct {
				Success bool            `json:"success"`
				Data    json.RawMessage `json:"data"`
			}
			if json.Unmarshal(body, &payload) != nil || !payload.Success ||
				len(payload.Data) == 0 || string(payload.Data) == "null" {
				return fmt.Errorf("%s returned an invalid public response", path)
			}
		}
	}
	return nil
}
