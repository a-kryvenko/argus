# Atmospheric density forecast

`GET /api/v1/public/forecasts/atmospheric-density` returns the latest internally
generated JB2008 forecast. It requires no body or model inputs and reads the
artifact produced by the application's forecast job.

The standard API envelope contains 49 hourly predictions (leads 0–48). Each
prediction contains latitude/altitude cells with longitude-mean density and
longitude p10/p90 in kg/m³. These percentiles describe spatial variation, not
uncertainty. The grid spans 200–800 km in 25 km steps and −90–90 degrees latitude
in 10 degree steps, averaging eight longitudes.

Run `python -m app.commands.generate_atmospheric_density_forecast` from
`apps/api`, or use the general `generate_forecast` command. Generation reads raw
`measurement` data, derives inputs in memory, and atomically writes the artifact
at `models.atmospheric_density.forecast_path`. It does not write derived metrics
to either shared observation table. No migrations or new observation collectors
are required. No input forecast CSVs or manually supplied snapshots are used.

## Solar backgrounds

Only `f10_7`, `s10`, `m10`, `y10`, `dst` and `ap` are queried, over 88 days.
The solar indices use 1/1/2/5-day lags. For each index, the last observation per
UTC day is selected, ending at the latest observation at or before its lagged
cutoff. The arithmetic average requires 81 consecutive valid calendar days;
days have equal weight regardless of sampling frequency. Missing calendar days beyond the short-gap policy below, invalid final daily
samples and stale inputs prevent publication.

The method is **trailing_81_daily_values**, an operational approximation to the
model's centered backgrounds. A true centered mean would require 40 subsequent
days of observations. No future observations or index predictions are used.
The derived columns are named `*_81mean`; the low-level model adapter passes
them into JB2008's background parameters. Values remain fixed over all 48 hours.

## DTC

**causal_dst_ap_v1** derives a temperature correction in kelvin from observed
hourly Dst and lagged 3-hour ap. Quiet-time correction is
`ap + 100 * (1 - exp(-0.08 * ap))`, with ap capped at 50 and a 0.279-day lag.
The storm equations use the reference main-phase coefficients, recovery
correction `0.13 * Dst`, and late-recovery slope −2.5, with nonnegative output.

This is a causal adaptation, not an exact reproduction of the retrospective
SET DTCFILE product. It uses the running observed storm minimum, recognizes
onset at Dst ≤ −75 with a drop ≥50, confirms recovery after three rising hours,
and detects late recovery from three shallow slopes. A magnitude-dependent
0/1/2-hour delay is applied. The reference DTCMAKEDR event detector needs future
Dst; this implementation never requests or synthesizes those future inputs.
The snapshot is held fixed for the forecast, including DTC.

DTC uses up to seven days of history, requires at least 72 consecutive hourly
Dst/ap samples and a six-hour quiet initialization interval. Dst must be no more
than three hours old. ap is selected as-of each lagged time with a maximum
three-hour hold. Dst gaps are not interpolated. Solar inputs must be within
48 hours of their lagged cutoffs. A fresh installation with less than 81 days
of solar observations still needs historical data before it can publish.

## Provenance and availability

Response metadata includes `background_method`, `dtc_method`, `history_start`,
`observed_at` (oldest of the latest selected driver observations), and
`dtc_observed_at`. The CSV additionally retains all derived input values.
API returns 503 for missing, malformed, incomplete or older-than-six-hours
artifacts. Existing artifacts without preparation metadata must be regenerated.

Sources:
- [JB2008 model paper, sections III and V](https://sol.spacenvironment.net/JB2008/pubs/AIAA_2008-6438_JB2008_Model.pdf)
- [Developer source archive, DTCMAKEDR_AUTO.f: DTCAP and DSTDTC](https://sol.spacenvironment.net/JB2008/downloadables/jb2008.zip)

The causal storm segmentation and trailing backgrounds have unit and integration
tests; their operational density error against authoritative DTC/density data has
not been calibrated in this change.


## Short gaps in the background window

Up to **two missing days in total per solar index** inside the 81-day window
are linearly interpolated between available neighboring daily values. Both
neighbors must already be available at the index's lagged cutoff. No leading
or trailing gaps, longer gaps, or explicitly invalid observed values are filled.
The current solar input, DTC and shared observations are unchanged.

When filling is used, `background_method` is
`trailing_81_daily_values_linear_gapfill`, and `background_interpolated_days`
lists the restored UTC dates per index, e.g.
`{"s10": ["2026-09-03", "2026-09-04"]}`. With complete observations the original
method is retained and this mapping is empty. The same metadata is saved in the
forecast CSV. These are estimated historical values used only in the mean,
not newly obtained observations or predictions of future inputs.
