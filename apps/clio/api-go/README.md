# Clio Go API

From the repository root, run `./argus compose up -d --build clio`. The dev API listens on `127.0.0.1:8001`.

For `go run .` use Go 1.25+, the five `CLIO_DB_*` connection variables, `OBSERVATIONS_SERVICE_TOKEN`, and `S3_PUBLIC_BASE_URL` for the SDO catalog. Apply the Clio migrations first. `CLIO_HTTP_ADDR` defaults to `:8000`.

Test with `go test ./...`. For PostgreSQL contract checks, set `TEST_DATABASE_ADMIN_DSN` to an isolated PostgreSQL server and run `apps/clio/.venv/bin/python -m pytest --rootdir=. --confcutdir=. --import-mode=importlib apps/clio/tests/integration/test_go_api.py` from the repository root.
