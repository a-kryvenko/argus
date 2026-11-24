package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"
)

func smoke(client *http.Client, base, expectedUpstream, token string) error {
	if token == "" {
		return errors.New("OBSERVATIONS_SERVICE_TOKEN is required for Clio smoke")
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
		if err != nil {
			return nil, err
		}
		if response.StatusCode != http.StatusOK {
			response.Body.Close()
			return nil, fmt.Errorf("clio: smoke endpoint %s returned %d", path, response.StatusCode)
		}
		return response, nil
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
			return errors.New("router did not select the expected Clio instance")
		case <-time.After(100 * time.Millisecond):
		}
	}
	response, err := request(context.Background(), "/internal/v1/observations/latest", token)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	var body struct {
		Success bool
		Data    struct {
			Points []json.RawMessage
		}
	}
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		return fmt.Errorf("decode latest observations: %w", err)
	}
	if !body.Success || body.Data.Points == nil {
		return errors.New("invalid latest observations")
	}
	return nil
}
