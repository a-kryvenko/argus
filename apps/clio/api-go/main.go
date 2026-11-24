package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	if len(args) != 0 {
		if len(args) != 3 || args[0] != "smoke" {
			return errors.New("usage: clio-api [smoke BASE EXPECTED_UPSTREAM]")
		}
		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.Proxy = nil
		client := &http.Client{
			Transport: transport,
			Timeout:   30 * time.Second,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
		if err := smoke(client, args[1], args[2], os.Getenv("OBSERVATIONS_SERVICE_TOKEN")); err != nil {
			return err
		}
		log.Print("[deploy] clio: smoke passed")
		return nil
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	dsn, err := databaseURL()
	if err != nil {
		return err
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return fmt.Errorf("configure PostgreSQL: %w", err)
	}
	defer pool.Close()

	addr := os.Getenv("CLIO_HTTP_ADDR")
	if addr == "" {
		addr = ":8000"
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	server := &http.Server{
		Handler:           newHandler(postgres{pool}, os.Getenv("OBSERVATIONS_SERVICE_TOKEN"), os.Getenv("S3_PUBLIC_BASE_URL")),
		ReadHeaderTimeout: 5 * time.Second,
	}
	serveErr := make(chan error, 1)
	go func() {
		serveErr <- server.Serve(listener)
	}()
	log.Printf("Clio listening on %s", listener.Addr())

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

func databaseURL() (string, error) {
	for _, name := range []string{"CLIO_DB_NAME", "CLIO_DB_USER", "CLIO_DB_PASSWORD", "CLIO_DB_HOST", "CLIO_DB_PORT"} {
		if os.Getenv(name) == "" {
			return "", fmt.Errorf("%s is required", name)
		}
	}

	database := os.Getenv("CLIO_DB_NAME")

	port, err := strconv.Atoi(os.Getenv("CLIO_DB_PORT"))
	if err != nil || port < 1 || port > 65535 {
		return "", errors.New("CLIO_DB_PORT must be an integer between 1 and 65535")
	}

	dsn := url.URL{
		Scheme:  "postgres",
		User:    url.UserPassword(os.Getenv("CLIO_DB_USER"), os.Getenv("CLIO_DB_PASSWORD")),
		Host:    net.JoinHostPort(os.Getenv("CLIO_DB_HOST"), strconv.Itoa(port)),
		Path:    "/" + database,
		RawPath: "/" + url.PathEscape(database),
	}
	return dsn.String(), nil
}
