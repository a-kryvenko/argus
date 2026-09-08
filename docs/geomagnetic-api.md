# Geomagnetic observations and live summary

## Collection

```bash
pnpm db:migrate
pnpm app:geomagnetic          # one pass
pnpm app:geomagnetic --watch  # independent source loops
```

Production runs the `geomagnetic` Compose service. Kp is polled every 60 seconds;
Dst every 300 seconds. Each source has a separate transaction and advisory lock.
Failures in one source do not stop the other's schedule. The service retries
failures; a one-pass command exits nonzero if either source fails.

Sources:

- [NOAA estimated planetary Kp](https://www.spaceweather.gov/products/planetary-k-index)
  via `https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json`.
- [WDC Kyoto Dst](https://wdc.kugi.kyoto-u.ac.jp/dstdir/)
  via `https://services.swpc.noaa.gov/products/kyoto-dst.json`.
- [NOAA JSON format change notice](https://www.weather.gov/media/notification/pdf_2026/scn26-21_Data_Format_Changes_Impacting_SWPC_Products.pdf)
  documents the current array-of-objects format.

Kp retains fractional values such as 3.33 in native three-hour UTC intervals. Dst
is an hourly value in nT. `time_tag` is retained as the interval start; interval
ends are start + 3h / 1h. Current-source Dst time tags align with the first hourly
value at 00:00 in the [Kyoto hourly table](https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/).

`geomagnetic_observation` is independent of the normalized model input tables.
It stores the original source record, nullable value, quality, interval boundaries
and receipt time, keyed by `(metric, interval_start)`. Identical responses are
no-ops. Corrections replace values and update receipt time. Receipt time is not
polling time. Previous revisions are not archived; data are not automatically deleted.
Recovery after downtime is bounded by the rolling history available in each feed.

Kp has `data_status=estimated`; Dst has `data_status=realtime`, not final or the
separate Kyoto provisional product. Both may be revised. `quality=unverified`
means Argus has not independently validated the value; `missing` marks null,
nonfinite, sentinel or invalid values. Kp with zero contributing stations is
`flagged`; its value is retained by the API but omitted from the graph. Original
fields, including station counts, are retained in storage.

## API

All paths below are relative to `/api/v1` and use `{success,data,error}`.

- `GET /public/observations/geomagnetic/latest`
- `GET /public/observations/geomagnetic/history?from=2026-09-07T00:00:00Z&to=2026-09-08T00:00:00Z`
- `GET /public/observations/summary`

Latest returns `generated_at` and `series.kp` / `series.dst`. Each series has
metadata, `latest`, `lag_seconds` and `status` (`missing`, `fresh`, `stale`). A
sample includes `interval_start`, `interval_end`, `interval_status`, `value`,
`quality`, `received_at` and nullable `station_count`. Missing newest values do
not fall back to older good ones. `interval_status=completed` describes elapsed
time only; it does not imply a final scientific value.

Freshness uses lag from interval **end**, clamped to zero for ongoing intervals.
The stale threshold allows the next native interval plus one hour for
publication: four hours for Kp, two for Dst. This is an Argus display policy, not a
provider delivery guarantee. Freshness and quality are independent.

History defaults to 24h and accepts positive ranges up to 31 days. Both dates
must include a timezone. It selects intervals overlapping `[from,to)` and preserves
their full boundaries. Adjacent queries may contain the same overlapping interval;
deduplicate on `metric, interval_start`. Records are oldest first, with no filling,
interpolation or conversion of Kp into hourly values. Empty storage returns empty
point lists and null latest samples, not zeros. HTTP failures remain errors.

## Summary calculations

The summary endpoint shares its calculations with the page and returns current
solar wind and geomagnetic series, `changes_1h` for speed, density, Bz and Bt, and
`southward_bz`. Each derived result has a status, nullable value, reason when
unavailable, and an `as_of` timestamp when available. These are observation
statistics, not predictions or local impact estimates.

An hourly change is the difference between two five-minute means whose endpoints
are one hour apart, anchored to that metric's latest sample. At least 4 of 5 valid
samples must exist in each window, and 48 of 60 in the last hour. Input older than
ten minutes, a missing/flagged latest sample, a spacecraft switch across the
comparison period, or inadequate coverage suppress the result. The API reports
all three coverage fractions and both means. No gap is interpolated.

Southward Bz counts consecutive valid one-minute negative samples from the same
spacecraft through its latest timestamp. A nonnegative preceding sample establishes
the onset; zero latest Bz produces zero duration. Missing/flagged minutes or a source
switch before an established onset make the result unavailable. If all available
consecutive samples are negative, `lower_bound` explicitly reports at least that
many sampled minutes; it does not invent an onset at the history boundary. The
lookback is bounded at 75 minutes. The timestamp and sampled-minute unit prevent
confusing this with a continuous real-time measurement through the current second.

The page displays Kp as three-hour blocks and Dst as hourly segments with UTC hover
and keyboard-focus details. Both share the selected 6h/24h/3d/7d/30d period with
solar wind. Missing/flagged intervals remain blank. Per-metric history
[coverage](observation-recovery.md#historical-coverage) counts usable native intervals.
