package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"gopkg.in/yaml.v3"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	token := os.Getenv("FORECASTS_SERVICE_TOKEN")
	if len(args) == 3 && args[0] == "smoke" {
		return smoke(args[1], args[2], token)
	}
	addr := envOr("PROPHET_HTTP_ADDR", ":8000")
	if len(args) != 0 {
		if args[0] != "serve" {
			return errors.New("usage: prophet-api [serve [--host HOST] [--port PORT] | smoke BASE EXPECTED_UPSTREAM]")
		}
		flags := flag.NewFlagSet("prophet-api serve", flag.ContinueOnError)
		host := flags.String("host", "0.0.0.0", "listen host")
		port := flags.Int("port", 8000, "listen port")
		if err := flags.Parse(args[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil
			}
			return err
		}
		if flags.NArg() != 0 || *port < 0 || *port > 65535 {
			return errors.New("serve accepts a host and a port from 0 to 65535")
		}
		addr = net.JoinHostPort(*host, strconv.Itoa(*port))
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	workdir, err := workDir()
	if err != nil {
		return err
	}
	maxAge, err := densityMaxAge(workdir)
	if err != nil {
		return err
	}
	dsn, err := databaseURL()
	if err != nil {
		return err
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return fmt.Errorf("configure Prophet PostgreSQL: %w", err)
	}
	defer pool.Close()

	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	server := &http.Server{
		Handler:           newHandler(postgres{pool}, token, filepath.Join(workdir, "data", "prophet", "summary", "latest.json"), maxAge),
		ReadHeaderTimeout: 5 * time.Second,
	}
	serveErr := make(chan error, 1)
	go func() { serveErr <- server.Serve(listener) }()
	log.Printf("Prophet listening on %s", listener.Addr())
	select {
	case err := <-serveErr:
		return err
	case <-ctx.Done():
		stop()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdown); err != nil {
			return errors.Join(err, server.Close())
		}
		return nil
	}
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func databaseURL() (string, error) {
	for _, name := range []string{"PROPHET_DB_NAME", "PROPHET_DB_USER", "PROPHET_DB_PASSWORD"} {
		if os.Getenv(name) == "" {
			return "", fmt.Errorf("%s is required", name)
		}
	}
	port, err := strconv.Atoi(envOr("PROPHET_DB_PORT", "5432"))
	if err != nil || port < 1 || port > 65535 {
		return "", errors.New("PROPHET_DB_PORT must be an integer between 1 and 65535")
	}
	database := os.Getenv("PROPHET_DB_NAME")
	dsn := url.URL{
		Scheme:   "postgres",
		User:     url.UserPassword(os.Getenv("PROPHET_DB_USER"), os.Getenv("PROPHET_DB_PASSWORD")),
		Host:     net.JoinHostPort(envOr("PROPHET_DB_HOST", "localhost"), strconv.Itoa(port)),
		Path:     "/" + database,
		RawPath:  "/" + url.PathEscape(database),
		RawQuery: "default_transaction_read_only=on&statement_timeout=10000&timezone=UTC",
	}
	return dsn.String(), nil
}

func workDir() (string, error) {
	if directory := os.Getenv("ARGUS_WORKDIR"); directory != "" {
		return directory, nil
	}
	directory, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(directory, "configs", "models_registry.yaml")); err == nil {
			return directory, nil
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			return "", errors.New("set ARGUS_WORKDIR to the application configuration directory")
		}
		directory = parent
	}
}

func densityMaxAge(directory string) (float64, error) {
	content, err := os.ReadFile(filepath.Join(directory, "configs", "models_registry.yaml"))
	if err != nil {
		return 0, err
	}
	var registry struct {
		Models struct {
			AtmosphericDensity struct {
				MaxAge *float64 `yaml:"max_age_hours"`
			} `yaml:"atmospheric_density"`
		} `yaml:"models"`
	}
	if err := yaml.Unmarshal([]byte(os.ExpandEnv(string(content))), &registry); err != nil {
		return 0, fmt.Errorf("read model registry: %w", err)
	}
	if maxAge := registry.Models.AtmosphericDensity.MaxAge; maxAge != nil {
		if math.IsNaN(*maxAge) || math.IsInf(*maxAge, 0) {
			return 0, errors.New("atmospheric_density.max_age_hours must be finite")
		}
		return *maxAge, nil
	}
	return 6, nil
}
