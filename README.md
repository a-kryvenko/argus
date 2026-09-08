# Argus Sunwatch

**Solar Activity Impact Forecasting & Decision Intelligence**


## Overview

[Argus Sunwatch](https://argussun.com/) is a software system designed to analyze solar activity data and provide solar activity forecast and **risk-oriented insights on potential impacts to terrestrial infrastructure**, including electrical power systems.


The project focuses on building a **decision-support framework** that combines real-time solar observations with historical event analysis to produce actionable risk indicators for operational awareness.

---

## Objectives

- Monitor real-time solar activity using publicly available data sources
- Analyze historical correlations between solar events and infrastructure disturbances
- Generate risk scores and alerts indicating potential impact levels
- Provide a foundation for infrastructure resilience and operational planning

---

## Key Features

- Real-time solar data ingestion
- Time-series processing and feature extraction
- Heuristic and evolving ML-based forecasting models
- Risk classification (Low / Medium / High)
- Alerting system for elevated solar activity
- API for integration with external systems
- Dashboard for visualization and monitoring

---


## System Architecture

1. Data Ingestion Layer
    1. Historical data sources:
        - [OMNIWeb](https://omniweb.gsfc.nasa.gov/) - provide Bx, By, Bz, V (Solar Wind), N (Density), T (Plasma temperature) in L1 Lagrange point.
        - [Solar Dynamics Observatory](https://data.nasa.gov/dataset/solar-dynamics-observatory) - provide solar observations. Data are loaded wia 
            [jsoc](https://jsoc1.stanford.edu/data/) (AIA, HMI)
        - [GONG](https://gong2.nso.edu/archive/patch.pl?menutype=zeroPoint#step2) - provide magnetic field map of the Sun in FITS format
    1. Live data sources:
        - [Deep Space Climate Observatory](https://epic.gsfc.nasa.gov/) - provide monitoring of Bx, By, Bz, V (Solar Wind), N (Density), T (Plasma temperature) in L1 Lagrange point. Data accessed wia [NOAA](https://services.swpc.noaa.gov/json/)
        - [Solar Dynamics Observatory](https://data.nasa.gov/dataset/solar-dynamics-observatory) - [jsoc](https://jsoc1.stanford.edu/data/) API is not reliable for live data, because of time delay up to 4 days. So data loaded as is directly from server
1. Processing Layer
    - Store source values in the narrow PostgreSQL `measurement` table
    - Materialize hourly, gap-filled values in the wide `normalized_observation` table
    - combine L1 sensors data with solar observations from [GONG](https://gong.nso.edu/)
1. Forecasting Layer
    - [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html), [LGBMRegressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html)
    - Machine learning models (finetunned [Surya](https://github.com/NASA-IMPACT/Surya), XGBoost), NYUAD Multimodal Encoder-Decoder, WSA-ENLIL / in-situ + empirical B
    - Risk scoring system
1. Impact Intelligence Layer (Private)
    - Correlation of solar events with power grid disturbances
    - Pattern recognition based on historical reports
    - Scenario-based risk estimation
1. Output Layer
    - Dashboard interface
    - Alerting system
    - REST API

---

## Example output

```JSON
{
    "valid_time": "2026-06-02T19:00:00.979417Z",
    "lead_hours": 1,
    "mean_v": 388.42530806514,
    "p_10_v": 375.187925180342,
    "p_50_v": 388.42530806514,
    "p_90_v": 397.381630752102,
    "prob_v_gt_450": 0.0863068688670829,
    "prob_v_gt_500": 0.0104562737642585,
    "prob_v_gt_600": 0.0025300442757748,
    "prob_v_gt_700": 0.0003081664098613,
    "kp_risk": 0
},
```

---

## Use Cases

- Situational awareness for infrastructure operators
- Research and analysis of space weather impact
- Decision-support for risk mitigation planning
- Integration into monitoring and alerting pipelines

---

## Disclaimer

This system is intended for research and decision-support purposes only.
It **should not be used as the sole basis for operational decisions** in critical infrastructure environments.

---

## Impact intelligence

- Power grid: input grid latitude + substation coords -> GIC risk.
- Satellite: input orbit altitude + inclination -> drag risk.
- GPS: simple TEC formula.
- Aviation: dose rate at FL350 or input route waypoints -> dose rate.

---

## Roadmap

- Expand real-time data integrations
- Improve forecasting models
- Enhance visualization dashboard
- Introduce anomaly detection
- Refine impact intelligence models
- Compare accuracy with [helioforecast](https://helioforecast.space/solarwind)

---

## Local development

PostgreSQL and Redis run in Docker, while the API and frontend run on the host
with their regular development servers and hot reload.

```bash
docker compose up -d
pnpm run dev
```

The `pnpm dev` command starts the API, frontend, minute solar-wind collector, and
Kp/Dst collector together. Apply `pnpm db:migrate` before the first start.

Check the infrastructure status or stop it with:

```bash
docker compose ps
docker compose down
```

The `postgres_data` and `redis_data` volumes preserve local data between
container restarts. Use `docker compose down --volumes` only when the local
database and Redis data should be deleted.

### Observation and forecast services

Minute solar wind observations have their own collector and API. Apply migrations
first, then run `pnpm app:solar-wind --watch` to collect continuously (omit `--watch`
for one pass). Deployment runs this as the dedicated `solar-wind` Compose service.
It polls the NOAA magnetic and plasma feeds independently every 60 seconds and
never runs hourly normalization or forecasts. See [solar wind API](docs/solar-wind-api.md).

Native Kp and Dst have an independent collector: `pnpm app:geomagnetic --watch`
(Kp every minute, Dst every five minutes). Production uses the `geomagnetic` Compose
service. The live page and `/public/observations/summary` share coverage-aware
observed trends. See [geomagnetic observations and summary](docs/geomagnetic-api.md).

Collection diagnostics are available on `/live` and `/public/observations/status`.
See [collector status and production healthchecks](docs/observation-status.md).

Observation ingestion and forecast generation are separate commands:

```bash
pnpm app:observations
pnpm app:forecast
```

Forecast commands read stored observations and never refresh them. Run ingestion
first on a new database. In deployment, observations refresh hourly at minute 0
and forecasts run at minute 10 using the latest committed data; these are independent
jobs, so the offset does not guarantee ingestion has finished. The ingestion command
also persists the longer solar history needed by atmospheric density forecasts.
Missing input data causes a forecast to fail or report that its product is unavailable.

Apply schema migrations with `pnpm db:migrate`. Observation endpoints expose
nullable S10, M10 and Y10 values when the configured backend supplies them.
Private model training, calibration, and generation instructions are maintained
in the private backend checkout.

See [package architecture](docs/architecture.md) for package boundaries,
private backend setup, and validation commands.

---

## Author

Andrii Kryvenko


Senior Software Engineer | Backend, Platform & Reliability

---

> Note: The commercial impact intelligence module (GIC risk assessment for power grids, aviation, satelites) is a proprietary closed-source component and is not included in this repository.
