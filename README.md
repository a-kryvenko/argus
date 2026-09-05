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

Check the infrastructure status or stop it with:

```bash
docker compose ps
docker compose down
```

The `postgres_data` and `redis_data` volumes preserve local data between
container restarts. Use `docker compose down --volumes` only when the local
database and Redis data should be deleted.

### Live S10, M10 and Y10 observations

The observations refresh also fetches GOES EUVS/XRS and applies a frozen
calibration from `models.solar_index_calibration.calibration_path` in
`configs/models_registry.yaml`.
Apply the schema migration with `pnpm db:migrate` before running the updated API.
The `/public/observations/latest` and `/public/observations/history` endpoints
expose nullable `s10`, `m10`, and `y10` fields, displayed on `/live`.

Run `notebooks/0_spacewx_data_fetching.ipynb` to fetch SOLFSMY reference files
for the registry datasets. It scans only `issue_time` from `data/training/2010_2024`
and `data/training/2025`, deduplicates observation dates across forecast horizons,
and writes separate references without changing the training shards.
Then prepare normalized historical GOES data at each dataset's `goes_path` and run
`notebooks/4_calibrate_solar_indices.ipynb`. This fits on the `train` dataset,
evaluates on the separate `validation` dataset, and saves the frozen calibration,
MAE/RMSE/bias, coverage, and validation estimates. All paths come from the same
registry entry used by live observations; no forecast models are trained.

The CLI `python -m forecast.data_pipelines.calibrate_solar_indices` remains
available for explicit `--goes`, `--solfsmy`, and `--output` paths; use the notebook
for registry-driven splitting and held-out validation.

GOES input uses the normalized columns returned by `fetch_goes`; SOLFSMY input
must contain `timestamp`, `s10`, `m10`, and `y10`. Both inputs may be CSV or
parquet. Historical training data and trained coefficients are not bundled;
the live GOES loader only retrieves the rolling live feed, not a training archive.
Validate a fitted calibration on held-out dates and relevant satellites before
using it operationally. Refreshes load the saved coefficients without retraining.

Each refresh persists the current hour boundary's provisional daily estimate,
using only quality-valid GOES records up to that boundary in the UTC day.
At midnight the estimate closes the previous day. The hour needs a valid sample
from the preceding hour; stale or missing GOES data produce no new estimate.
Earlier snapshots are retained rather than recomputed from a truncated live feed.
These estimates accumulate in `measurement` and `normalized_observation` as
refreshes run; they are not interpolated or carried into missing hours.
Without a calibration artifact, or if GOES fails, the other observations keep
updating, new index fields are `null`, and the API logs the reason.
No S10/M10/Y10 forecast products or forecast targets are added.

---

## Author

Andrii Kryvenko


Senior Software Engineer | Backend, Platform & Reliability

---

> Note: The commercial impact intelligence module (GIC risk assessment for power grids, aviation, satelites) is a proprietary closed-source component and is not included in this repository.
