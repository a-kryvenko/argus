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

---

> Argus Sunwatch is an independently designed and developed project. I am responsible for the system architecture, scientific data pipelines, forecasting methodology and experiments, backend services, APIs, infrastructure, deployment, and web application.
