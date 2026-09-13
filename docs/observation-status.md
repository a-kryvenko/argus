# Observation collection status

For restart behavior and its regression check, see [history recovery](observation-recovery.md).

`GET /public/observations/status` returns the standard API envelope with
`generated_at`, aggregate `status` (`ok` or `degraded`), and `sources` keyed by
`solar_wind_mag`, `solar_wind_plasma`, `kp`, and `dst`. Responses are not cached.
The expandable **Data collection** block on `/live` refreshes every minute.

Status storage was added in migration `20260908_0006`. A source with no recorded
attempt has status `not_started`; earlier collection outcomes are not reconstructed.

Each source exposes its polling interval, freshness thresholds, last attempt,
completion, valid parsed response, successful save, latest source measurement,
last error, and consecutive failure count. Kp and Dst also expose interval end.
Errors use fixed public messages; raw exceptions remain in server logs.
The last error is retained after recovery; zero consecutive failures means it
is historical. This table stores current diagnostics, not an event log.

`collector_status` and `data_status` are independent. The combined `status`
prioritizes the following conditions:

| Status | Meaning |
| --- | --- |
| `collector_stalled` | An attempt has remained unfinished for over 120 seconds |
| `collector_overdue` | No attempt for over two polling intervals plus 60 seconds |
| `collection_error` | The most recent completed attempt failed |
| `collecting` | Initial collection is in progress |
| `source_delayed` | Latest source measurement is outside its freshness window |
| `data_unavailable` | Latest response contains no usable current measurements |
| `data_partial` | Latest source measurement is incomplete |
| `ok` | Collection and source data are within their expected windows |

Solar wind and Kp are polled every 60 seconds; Dst every 300 seconds.
Solar wind freshness uses measurement time and a 600-second threshold.
Kp and Dst freshness uses interval end, with thresholds of 14,400 and 7,200
seconds respectively. Repeated retrieval of the same data updates collection
diagnostics but does not make the measurements fresher.

Attempts are committed before fetching. Success is committed atomically with
observations. Failures are saved in a separate transaction after observation
rollback. A parsed response may therefore have a newer measurement timestamp
even when its save failed; consult `last_success_at` and the error fields.
Overlapping source runs skipped by the advisory lock do not create attempts.
A database outage cannot be recorded in that same unavailable database; consult
collector logs and container health as well.

## Production liveness

Both collector services have a Compose healthcheck, every 30 seconds, with a
120-second startup period and three retries. From `apps/clio`, run a probe with:

```sh
.venv/bin/clio check-health solar-wind
.venv/bin/clio check-health geomagnetic
```

The probe reads an atomic heartbeat file local to the container and exits 0 for
healthy or 1 for unhealthy. It detects missing, stopped, overdue, or unfinished
loops independently of the database. An upstream request error alone does not
fail liveness if the collector keeps trying; it appears in the status API.
One-shot collections do not write watch-process heartbeats.

Docker marks an unhealthy container but does not automatically restart it for
that reason. The existing restart policy handles process exits. API statuses
infer overdue or interrupted collection from timestamps; they do not inspect
production processes or prove a particular process has stopped.
