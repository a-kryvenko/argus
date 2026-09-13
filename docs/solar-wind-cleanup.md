# Solar wind raw-history cleanup

Apply migration `20260908_0009` before using the command.

```sh
pnpm app:cleanup-solar-wind                 # report only
pnpm app:cleanup-solar-wind --apply         # delete verified raw hours
pnpm app:cleanup-solar-wind --json          # structured report
```

Only complete source hours older than 90 days are candidates. `--retention-days`
can increase that period, but cannot reduce it below 90. The cutoff rounds down
to a UTC hour so a partially expired hour remains intact. All spacecraft records
in a verified hour are deleted; five-minute/hourly aggregates and Kp/Dst remain.

The command examines at most 24 source hours per run (`--limit`, range 1–240).
Repeat to process more. Skipped hours remain candidates; use `--from` with a whole
UTC hour to inspect later data while resolving earlier problems.

## Verification and transactions

Each source hour must have all twelve five-minute aggregates and its hourly
aggregate, at the current version. Their statistics are recomputed from retained
raw data and compared using the audit's float tolerance. Missing/mismatched
aggregates or a pending recalculation skip the hour.

Report mode uses read-only snapshots and changes nothing. Apply mode obtains
short table locks covering raw observations, aggregates and pending jobs. If a
writer is active, `NOWAIT` skips the candidate. The locks prevent changes between
verification and deletion; new writers may wait until that hour's transaction
finishes. Each hour commits separately. A later failure does not undo hours already
committed; successful deletions are logged.

`solar_wind_retired_hour` records the kind, UTC hour, cleanup time and deleted row
count. This marker and deletion commit together. The queue trigger ignores these
cleanup deletes; ordinary raw deletions still request recalculation.

Database triggers reject raw inserts/updates and queue entries for retired hours.
This prevents later partial backfills or rebuilds from overwriting preserved
aggregates. Reopening a retired hour requires a separate restoration procedure
with complete raw data; no automatic restoration command is provided. Migration
downgrade refuses to remove this protection once any hour has been retired.

## Results

Reports list eligible/deleted row counts and skipped hours with reasons. Exit 1
means hours were skipped; exit 0 means the batch completed without skips, including
an empty batch. Database errors also return nonzero.

No cleanup cron job is installed. Deletion does not guarantee an immediate decrease
in database files; PostgreSQL can reuse the freed space. The audit can no longer
compare retired hours against raw data and reports them as unverifiable.

## Validation

```sh
./scripts/test-python -q
PYTHONPATH=apps/clio/src apps/clio/.venv/bin/python apps/clio/tests/integration/verify_solar_wind_retention.py
```

The integration check deletes synthetic records in a disposable schema. It covers
report/apply, cutoff boundaries, missing/mismatched/pending skips, writer contention,
rollback, preserved aggregates, repeated runs and protection against backfill.
