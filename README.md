# Argus Sunwatch

[Argus Sunwatch](https://argussun.com/) is a space-weather monitoring and
forecasting platform. It brings together solar-wind measurements, geomagnetic
observations and forecasts to help users follow current conditions and explore
how they change over time.

## What you can explore

- **Live solar wind:** NOAA measurements of the magnetic field,
  speed, density and temperature.
- **Geomagnetic activity:** three-hour Kp and hourly Dst observations.
- **Historical observations:** solar-wind time series, five-minute and hourly
  summaries, with information about data coverage and collection status.
- **Forecasts:** published model outputs alongside observational data.
- **Metrics:** accuracy of each forecast.

## Observations sources

| Source | Observations used by Argus |
|---|---|
| **NOAA SWPC** | Live solar-wind speed, proton density and temperature, and IMF components from the real-time solar-wind feeds; estimated Kp, Ap derived from Kp, Kyoto Dst distributed by SWPC, and F10.7 radio flux. |
| **NASA OMNIWeb / OMNI 2** | Historical hourly solar-wind plasma, magnetic-field and geomagnetic data for backfilling and model evaluation. |
| **ACE/SWEPAM via NASA SPDF; SOHO/CELIAS Proton Monitor** | Additional historical plasma sources, aggregated to hourly values when filling gaps. |
| **GFZ** | Historical F10.7 radio flux. |
| **SDO via Stanford JSOC** | Numerical near-real-time FITS: AIA 94, 131, 171, 193, 211, 304, 335 and 1600 Å, plus HMI line-of-sight magnetograms. PROSWIN uses the original AIA 171/211 Å pair. |
| **NSO GONG** | Solar magnetograms from the SWPC live feed and NSO archive, used to derive magnetic-field features for southward IMF forecasting. |
| **NOAA GOES** | Live EUV and X-ray background measurements from SWPC, with historical files from the NOAA archive. These support calibrated estimates of S10, M10 and Y10; those estimates are derived indices, not direct observations of the indices. |
| **SILSO** | Monthly sunspot numbers used as additional PROSWIN inputs. |

Clio collects and stores the configured observation feeds. The PROSWIN process
also maintains its own SILSO snapshot. Source priority, collection intervals and
backfill windows are defined in [the project configuration](configs/project.yaml).

## Forecasting

The configured forecasting pipeline uses the following models:

| Forecast | Model and inputs | Published output |
|---|---|---|
| **Solar-wind speed (`v`)** | **DLinear + PROSWIN**: hourly speed history combined with the official PROSWIN fold-1 model using AIA 171/211 Å images and physical features. Blend weights depend on lead time and were fitted on 2025 data. | Central speed forecast, q10/q90 intervals and speed-threshold probabilities. Missing PROSWIN predictions fall back to DLinear. |
| **Proton density (`n`)** | **DLinear + LightGBM**, using hourly n/v history and the native PROSWIN **speed forecast** as an additional input. | q10/q50/q90 density forecast.
| **Proton temperature (`T`)** | **LightGBM** with hourly T/n/v history and the PROSWIN speed forecast as an input (`argus-plasma-temperature-proswin-v1`). | Hourly q10/q50/q90 in K over +1…+96 h; without PROSWIN uses T/n/v history, then T-only if n/v is unavailable. |
| **Kp** | Calibrated classification models using solar-wind and geomagnetic history (`argus-kp-t-v2`). | Probabilities of exceeding Kp thresholds. |
| **Ap and Dst** | **LightGBM quantile regression**, using observation-history features (`argus-ap-q-v2`, `argus-dst-q-v2`). | q10/q50/q90 forecasts. |
| **Total IMF (`Bt`)** | **LightGBM classifiers with logistic probability calibration**, using hourly IMF/plasma history (`argus-bt-t-v3`). | Probabilities of Bt ≥ 5, 10 and 15 nT over +1…+24 hours. |
| **Southward IMF (`Bs`)** | Calibrated classification models using observation history and GONG magnetic features (`argus-bs-t-v2`). | Threshold probabilities for `Bs = max(−Bz, 0)`, rather than a signed Bz point forecast. |
| **Thermospheric density** | **JB2008**, driven by F10.7, calibrated S10/M10/Y10 and the geomagnetic temperature correction. Current operation holds observed drivers constant over the forecast horizon. | Atmospheric-density grids by time, altitude and location. |

Prophet schedules forecasts hourly and runs heavy calculations sequentially.
Before generating speed, density or temperature, it requests PROSWIN through a shared job
queue and waits for completion or a timeout. PROSWIN runs in a temporary CPU
process and releases model memory when the task finishes. Cached predictions are
shared between these products; the neural model loads only when new inference is needed.

The speed blend's prediction intervals and threshold probabilities currently use
DLinear residual distributions around the blended point forecast; they have not
yet been calibrated separately for the blend. Signed-Bz point forecasts are not
part of the current scheduled products. The solar-index
models listed in the registry are not separate scheduled radiation forecasts.

See the [model registry](configs/models_registry.yaml) and
[accuracy report](docs/accuracy/README.md) for configuration and evaluation scope.

## About this repository

This repository contains the web application, public API and services that collect
observations, prepare datasets and publish forecasts. Forecast model implementation
and training live in a separate private repository; generating forecasts requires
that backend and its model files.

## Explore the project

- [Open Argus Sunwatch](https://argussun.com/)
- [Usage help](https://argussun.com/help)
- [API documentation](https://argussun.com/api/v1/docs)
- [Development, configuration and deployment](README_DEPLOY.md)
- [Architecture and data flow diagrams](docs/architecture/README.md)
- [Forecast verification, model evaluation and MLflow](docs/forecast-workflows.md)
- [Forecast accuracy report](docs/accuracy/README.md)
- [Historical demo: event replay and dataset generation](docs/demo-mode.md)

---

> Argus Sunwatch is an independently designed and developed project. I am responsible for the system architecture, scientific data pipelines, forecasting methodology and experiments, backend services, APIs, infrastructure, deployment, and web application.
