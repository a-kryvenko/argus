package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/a-kryvenko/argus/apps/prophet/api-go/publicforecast"
)

func smoke(base, expectedUpstream, token string) error {
	if token == "" {
		return errors.New("FORECASTS_SERVICE_TOKEN is required for Prophet smoke")
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	client := &http.Client{
		Transport: transport, Timeout: 30 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	request := func(ctx context.Context, path, token string) (*http.Response, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+path, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", "Argus-Deploy/1")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		response, err := client.Do(req)
		if err == nil && response.StatusCode != http.StatusOK {
			response.Body.Close()
			return nil, fmt.Errorf("Prophet smoke endpoint %s returned %d", path, response.StatusCode)
		}
		return response, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for {
		response, err := request(ctx, "/health/ready", "")
		if err != nil {
			return err
		}
		response.Body.Close()
		if expectedUpstream == "" || response.Header.Get("X-Argus-Upstream") == expectedUpstream {
			break
		}
		select {
		case <-ctx.Done():
			return errors.New("router did not select the expected Prophet instance")
		case <-time.After(100 * time.Millisecond):
		}
	}
	const product = "solar-wind-speed"
	response, err := request(context.Background(), "/internal/v1/forecasts/"+product+"/latest", token)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	var data release
	if err := json.NewDecoder(response.Body).Decode(&data); err != nil {
		return fmt.Errorf("decode forecast release: %w", err)
	}
	if data.Product != product || data.ID == "" || len(data.Artifacts) != len(productArtifacts(product)) {
		return errors.New("invalid forecast release")
	}
	if err := checkPreparedForecast(request, product, token); err != nil {
		return err
	}
	log.Print("[deploy] prophet-api: smoke passed")
	return nil
}

func checkPreparedForecast(request func(context.Context, string, string) (*http.Response, error), product, token string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	for {
		response, err := request(ctx, preparedForecastPath+product, token)
		if err == nil {
			var forecast publicforecast.Forecast
			err = json.NewDecoder(response.Body).Decode(&forecast)
			response.Body.Close()
			if err != nil || forecast.Target != product || len(forecast.Predictions) == 0 {
				return errors.New("invalid prepared forecast")
			}
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("prepared forecast did not become available: %w", err)
		case <-time.After(100 * time.Millisecond):
		}
	}
}
