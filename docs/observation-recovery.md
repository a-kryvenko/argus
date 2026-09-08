# Recovery after collector downtime

Every collection replays the complete response from its configured source.
There is no latest-timestamp cutoff: records older than the latest stored point
can fill internal gaps or replace revised measurements. The same behavior applies
to startup, scheduled watch cycles and one-shot collection commands.

Solar wind rows are unique by kind, measurement time and spacecraft. Kp and Dst
rows are unique by metric and interval start. Repeated unchanged source records
do not update `received_at`; changed raw records replace the saved values and
receipt time. Each source response is committed atomically.

After a restart, the next successful collection restores all records still present
in the source response. Records already stored outside that response remain in
the database. Missing source records are not interpolated or invented. If a source
publishes a missing record later, a subsequent collection saves it.

Recovery is limited to the upstream response window. There is currently no archive
backfill for observations that disappeared from that window during a long outage.
An `ok` collection status describes current collection and measurement freshness;
it does not certify completeness of historical data. Coverage is reported below;
archive recovery and automatic deletion are not implemented.

## PostgreSQL regression check

From the repository root, with local database settings in `.env` / `.env.local`:

```sh
PYTHONPATH=apps/api apps/api/.venv/bin/python apps/api/tests/integration/verify_observation_recovery.py
```

The check uses synthetic source responses through the real ingestion functions
and PostgreSQL writes. It creates a uniquely named temporary schema and removes
only that schema on completion. It does not contact observation providers or
modify application observations. The database role needs schema creation rights.

For each of magnetic field, plasma, Kp and Dst it verifies recovery of internal
gaps, preservation of older data, repeated collection without duplicates or receipt
time changes, late publication, and revision of a previously saved value. Solar
wind fixtures span multiple SQL batches and an eight-hour missing interval.

## Historical coverage

Both history APIs include `coverage` for each metric. Live displays the percentage
of usable native slots and expandable missing/unusable ranges for the selected
period. `expected_slots`, `usable_slots`, `missing_slots`, and `invalid_slots`
count slots, not interpolated values. Null values and provider-flagged samples
are unusable. `percent` is null when no slots are expected.

Solar wind counts minute timestamps in `[from,to)` (the first timestamp is rounded
up to a UTC minute). Kp/Dst count native UTC intervals overlapping that range;
partial edge intervals each count once. Percentages measure sample/interval
availability, not duration-weighted coverage. Future time is excluded using
`evaluated_to`. An in-progress geomagnetic interval can contribute to coverage;
this does not mean it is final or independently validated.

`gaps` contains contiguous ranges with the same `reason` (`missing` or `invalid`),
clipped to the requested range, with native slot counts. Live displays the first
50 ranges; the API returns the full list. Missing recent records may reflect
normal publication delay. Coverage cannot distinguish an upstream omission from
collector downtime and does not label either as the cause.

Aggregate history has a separate contract: coverage describes available calculated
windows, while `processing` describes whether calculation is available or pending.
See [aggregate history semantics](solar-wind-aggregation.md). Missing aggregate
results do not count as missing raw measurements.
