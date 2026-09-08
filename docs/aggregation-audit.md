# Read-only aggregation and retention audit

Run `pnpm app:audit-solar-wind` against the configured database. It checks the
last seven days ending at the current UTC hour. Use explicit whole UTC hour
boundaries to inspect another period (at most 31 days per run):

```sh
pnpm app:audit-solar-wind --from 2026-09-01T00:00:00Z --to 2026-09-08T00:00:00Z
pnpm app:audit-solar-wind --json --detail-limit 1000
```

The command runs in a PostgreSQL `REPEATABLE READ READ ONLY` transaction. Concurrent
ingestion or recalculation cannot mix different committed states within a report.
It neither fetches sources, recalculates/saves aggregates, changes the pending
queue nor deletes data.

For each source hour occupied by raw records, aggregates or queued work, the
command compares twelve five-minute and one hourly bucket. It uses retained raw
observations and the current aggregation rule, comparing all statistics including
count, sum, mean, min/max, coverage, first/last values, source transitions and Bz
negative count. The aggregation version is checked too. Float means permit
rounding noise (relative/absolute tolerance `1e-9`); integer counts must match.

The report distinguishes missing aggregates, differing field paths, pending source
hours and buckets that cannot be verified because the entire source hour has no
retained raw observations. Hours with no raw records, aggregates or queued work
are outside comparison scope: this audit does not establish upstream completeness.
Recomputing with the current rule verifies persistence consistency, not independent
scientific validation of that rule. Unit fixtures test its arithmetic separately.

Issue totals are complete; details default to the first 200 entries. Increase
`--detail-limit` to see more. Exit code 1 means issues were found; 0 means `ok` or
`no_data` (inspect status to distinguish them). Database/command failures also
return nonzero and do not produce a successful audit report.

The storage section covers the entire solar wind raw table, independently of the
comparison range. It reports rows older than `--retention-days` (default 90),
active rows among them, summed stored row payload bytes, and their estimated share
of total allocated table/index space. This proportional estimate is not disk space
guaranteed to be freed by deletion. It excludes other tables, WAL and backups.
These counts are age candidates, not records certified for cleanup.

The storage scan reads the whole table; run off-peak as it grows. Individual
statements have a 60-second timeout and raw comparison is loaded one source hour
at a time. Split large date ranges into adjacent runs. A report is only a snapshot:
future cleanup must revalidate against revisions and pending work at deletion time.

Validation:

```sh
./scripts/test-python -q
PYTHONPATH=apps/api apps/api/.venv/bin/python apps/api/tests/integration/verify_solar_wind_aggregation.py
```

The integration fixture checks known missing/corrupted aggregates, absent raw
data, old-row estimates, issue truncation and PostgreSQL transaction settings in
a disposable schema, without changing application observations.
