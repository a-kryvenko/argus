# Argus Sunwatch

[Argus Sunwatch](https://argussun.com/) is a space-weather monitoring and
forecasting platform. It brings together solar-wind measurements, geomagnetic
observations and forecasts to help users follow current conditions and explore
how they change over time.

The project connects the full journey from collecting observations to publishing
forecasts, with a web interface for exploring the data and an API for using it
in other applications and analyses.

## What you can explore

- **Live solar wind:** minute-by-minute NOAA measurements of the magnetic field,
  speed, density and temperature.
- **Geomagnetic activity:** three-hour Kp and hourly Dst observations.
- **Historical observations:** solar-wind time series, five-minute and hourly
  summaries, with information about data coverage and collection status.
- **Forecasts:** published model outputs alongside observational data. Available
  products depend on the configured models and input data.

The `/live` observation workspace combines a selectable measurement summary,
solar-wind and geomagnetic charts with a shared UTC history window, and an
inspector for source, quality, freshness and coverage. The inspector collapses on
smaller screens. Collection diagnostics and additional hourly indices remain
available below the charts.

The forecast overview (`/`), product catalog (`/products`) and product pages share
the observation workspace's navigation and graphite theme. Forecasts provide
variable selection, 24/48-hour or full-release views, median and q10–q90 charts,
threshold probability heatmaps, and an inspector with exact UTC times and values.
The hourly table also supports keyboard selection. Missing values remain gaps;
changing the visible horizon never changes the model's issue time.

The model performance workspace (`/metrics`) uses the same navigation and layout.
Product pages connect continuous scores, threshold scores and calibration through
a shared lead-hour inspector, with metric and variable selectors, horizon controls
and accessible value tables. Charts use the actual evaluated lead hours and retain
gaps and unavailable scores instead of shifting values or replacing them with zero.


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
- [Forecast accuracy report](docs/forecast-accuracy-report.md)
- [Historical demo: event replay and dataset generation](docs/demo-mode.md)

---

> Argus Sunwatch is an independently designed and developed project. I am responsible for the system architecture, scientific data pipelines, forecasting methodology and experiments, backend services, APIs, infrastructure, deployment, and web application.
