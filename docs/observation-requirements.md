# Observation requirements

Repository configuration as of October 10, 2026. Live data coverage is not verified.

## Data to collect

History below is the configured backfill depth, except where stated otherwise.

| Data | Sources | Model sampling | History |
| --- | --- | --- | --- |
| IMF Bx/By/Bz, nT | SWPC RTSW; OMNI archive | Hourly, retain raw samples | 60 days |
| Solar-wind speed v, km/s; proton density n, cm⁻³; temperature t, K | SWPC RTSW; OMNI/ACE/SOHO archives | Hourly, retain raw samples | 60 days |
| Kp, ap | SWPC; OMNI archive | Native 3-hour intervals | 60 days |
| Dst, nT | SWPC; OMNI archive | Hourly | 60 days |
| F10.7, sfu | SWPC; GFZ/OMNI archives | Daily | 88 days |
| GOES EUV, Lyman-alpha, Mg II, X-ray background, quality and Earth–Sun correction | GOES live/archive | Timestamped originals → daily solar indices | 88 days |
| GONG magnetograms | GONG live/archive | Hourly collection | 2 days; freshness limit 3 hours |
| AIA 171/211 Å image pairs | Clio SDO archive | Hourly slots | PROSWIN target = image time +96 hours |
| Monthly sunspot number | SILSO, fetched by PROSWIN | Monthly | 12 consecutive available months required |

Homepage AIA images: `clio collect sdo-browse` runs hourly when `clio.sdo_browse_enabled` is true. Keeps two snapshots per channel in `data/observations/sdo_browse`; JPEG/WebP at 128–4096 px. `clio cleanup` also removes older variants.

S10/M10/Y10 are calibrated GOES estimates. SOLFSMY supplies reference values for
calibration and validation. The GOES parser requires irradiance channels
256/284/304/1175/1216/1335/1405 Å plus the fields listed above.
AIA 193 Å supports the older Ridge feature path; the current plasma adapter uses PROSWIN.

## Model inputs

**Shared hourly frame:** Bx/By/Bz/v/n/t/Kp/ap/Dst/F10.7, with calibrated
S10/M10/Y10 where available. Exact feature subsets come from each model artifact.

| Registry entry / model | Required inputs |
| --- | --- |
| `plasma_speed_dlinear` | Observed hourly v; 57 days of context plus fill buffer |
| `plasma_speed_quantile`, `plasma_speed_threshold` | DLinear v history; optional PROSWIN speed |
| `plasma_density_quantile` | Observed hourly n; recent n/v and optional PROSWIN speed for blend heads |
| `plasma_temperature_quantile` | Hourly t over a 48-hour window, latest ≤2 hours old; n/v and PROSWIN for richer heads |
| PROSWIN | AIA 171/211 pairs; 49 hourly v values at target −672…−624 hours; 12 sunspot months |
| `kp_threshold`, `ap_quantile`, `dst_quantile` | Shared hourly frame |
| `hmf_total_threshold` | 168 completed, unfilled hourly bins of Bx/By/Bz/v/n/t, with coverage metadata |
| `hmf_southward_threshold` | Shared hourly frame + GONG latitude-band features |
| `f10_7_quantile`, `s10_quantile`, `m10_quantile`, `y10_quantile` | Shared hourly frame enriched with solar-index estimates |
| `solar_index_calibration` | GOES + calibration artifact; paired SOLFSMY values for training |
| `dtc` | Hourly Dst + 3-hour ap; DTC is calculated |
| `atmospheric_density` | JB2008 or DTM2020 inputs below |

Keep observed hourly v/n/t separate from interpolated data. DLinear history
windows and allowed filling are defined by the artifact. The alternative boosted
Kp service uses the same sources, including a 168-hour feature window.

## JB2008 and DTM2020

Requirements of the implemented adapters:

| Input | JB2008 | DTM2020 Operational |
| --- | --- | --- |
| Solar indices | F10.7, S10, M10, Y10 | F10.7 |
| Solar lag | F10.7/S10: 1 day; M10: 2 days; Y10: 5 days | F10.7 at issue −1 day |
| Background | 81 daily values per index, ending at its lagged cutoff | 81 daily F10.7 values, ending at latest available day |
| Solar freshness | Latest sample ≤48 hours before lagged cutoff | Latest F10.7 ≤48 hours old; interpolation hold ≤1 day |
| Geomagnetic history | Up to 168 Dst hours; consecutive valid tail ≥72 hours, including 6 quiet hours for initialization; ap also covers the 6.696-hour lag | Last 30 hours of ap; ≥8 consecutive closed 3-hour bins with coverage for interpolation |
| Geomagnetic freshness | Latest Dst ≤3 hours old; lagged ap lookup tolerance 3 hours | ap hold ≤6 hours |
| Derived driver | Causal DTC from Dst/ap | Kp from ap: −3-hour value and mean of 24 hourly values |

Both use trailing 81-day means. At most two missing interior days may be filled
per background series; no edge extrapolation or invalid-value replacement.
DTM2020 does not require S10/M10/Y10, GOES, Dst or DTC.

**Current configuration:** JB2008. Both adapters hold observed drivers fixed over
0–48 hours; forecast ap is not enabled. DTM2020 additionally needs its native
runtime, coefficients and suitable license. Density releases expire after 6 hours.

**Density advisory inputs:** spacecraft apoapsis, periapsis and inclination;
reference comes from the published density model background. The experimental loss calculator additionally needs circular-orbit
altitude/inclination, mass, effective area and drag coefficient.

## Observations for accuracy checks

| Output | Validation data |
| --- | --- |
| Plasma, IMF, Kp/ap/Dst | Corresponding observed values at forecast target times, with quality and revision |
| Solar indices | Reference F10.7 and SOLFSMY S10/M10/Y10 |
| Atmospheric density | Satellite neutral density, kg/m³; UTC, altitude, latitude/longitude, product version, quality and uncertainty |
| Altitude loss / drag | Precise orbit and velocity histories; mass, area/attitude, drag coefficient and maneuver records; accelerometer data where available |

Current density validation: GRACE-FO 1, 42 days in 2025, approximately 452–503 km.
Actual orbital loss is not validated. Broader validation needs multiple satellites,
altitudes and storms, with separate calibration and test periods.

## Storage and known gaps

Store value/unit, source/product, observation interval, receipt time, quality and
revision. Keep original files and checksums, calibration versions and issued
forecasts. Forecast inputs must have been available at issue time.

- JB driver preparation does not enforce `available_at`; DTM does.
- Speed/density input DTOs lack per-point receipt times; HTTP reads are not durable historical snapshots.
- SILSO historical availability is assumed; PROSWIN fetches it outside Clio.
- Configured backfill does not establish complete 81-day solar-index coverage.

Sources: [registry](../configs/models_registry.yaml), [collection settings](../configs/project.yaml),
[input contract](../packages/forecast/src/forecast/observation_inputs.py),
[JB2008 drivers](../packages/forecast-core/src/forecast_core/models/jb2008_drivers.py),
[DTM2020 drivers](../packages/forecast-core/src/forecast_core/models/dtm2020_drivers.py),
[accuracy](accuracy/ATMOSPHERIC.md).
