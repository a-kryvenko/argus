package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

type upstream struct{ url, token string }

type api struct {
	client       *http.Client
	clio         upstream
	prophet      upstream
	intelligence upstream
	freshness    map[string]time.Duration
}

func main() {
	if len(os.Args) == 4 && os.Args[1] == "smoke" {
		if err := smoke(os.Args[2], os.Args[3]); err != nil {
			log.Fatal(err)
		}
		return
	}
	app, err := loadAPI(os.Getenv("ARGUS_WORKDIR"))
	if err != nil {
		log.Fatal(err)
	}
	address := os.Getenv("PUBLIC_API_HTTP_ADDR")
	if address == "" {
		address = ":8000"
	}
	listener, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatal(err)
	}
	server := &http.Server{Handler: app.routes(), ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second, WriteTimeout: 115 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	serverErr := make(chan error, 1)
	go func() { serverErr <- server.Serve(listener) }()
	log.Printf("Public API listening on %s", listener.Addr())
	select {
	case err := <-serverErr:
		if !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	case <-ctx.Done():
		stop()
		shutdown, cancel := context.WithTimeout(context.Background(), 110*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdown); err != nil {
			log.Print(err)
			server.Close()
		}
	}
}

func loadAPI(workdir string) (*api, error) {
	if workdir == "" {
		workdir = "."
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.DialContext = (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	app := &api{
		client: &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
		clio:   configuredUpstream("OBSERVATIONS"), prophet: configuredUpstream("FORECASTS"),
		intelligence: configuredUpstream("INTELLIGENCE"),
		freshness:    make(map[string]time.Duration),
	}
	var project struct {
		Clio struct {
			Observations map[string]struct {
				FreshnessMinutes float64 `yaml:"freshness_minutes"`
			} `yaml:"observations"`
		} `yaml:"clio"`
	}
	if err := readYAML(filepath.Join(workdir, "configs", "project.yaml"), &project); err != nil {
		return nil, err
	}
	for metric, policy := range project.Clio.Observations {
		app.freshness[metric] = time.Duration(policy.FreshnessMinutes * float64(time.Minute))
	}
	app.freshness["bt"] = min(app.freshness["bx"], app.freshness["by"], app.freshness["bz"])
	return app, nil
}

func configuredUpstream(name string) upstream {
	return upstream{strings.TrimRight(os.Getenv(name+"_URL"), "/"), os.Getenv(name + "_SERVICE_TOKEN")}
}

func readYAML(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := yaml.Unmarshal([]byte(os.ExpandEnv(string(data))), target); err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	return nil
}
