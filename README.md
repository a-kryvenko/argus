# Argus Sunwatch

[Space-weather observations, forecasts and satellite density-risk advisories](https://argussun.com/).

| Service | Responsibility |
| --- | --- |
| [Clio](apps/clio/README.md) | Collect and archive observations |
| [Prophet](apps/prophet/README.md) | Publish forecasts and density backgrounds |
| [Intelligence](apps/intelligence/README.md) | Assess density exposure on supplied ephemerides |
| [API](apps/api/README.md) / [Web](apps/web/README.md) | Client access and dashboard |

Clio → Prophet → Intelligence exchange data over HTTP and own separate databases.
Scientific backends live in private `forecast-core` and `intelligence-core` repositories.

- [Setup and deployment](README_DEPLOY.md) · [Commands](docs/commands.md)
- [Architecture](docs/architecture.md) · [Dataflow diagrams](docs/architecture/README.md)
- [Observation requirements](docs/observation-requirements.md)
- [Forecast accuracy](docs/accuracy/README.md) · [Solar events](docs/solar-wind-events.md)
- [Density advisories](docs/density-advisory.md) · [Historical demo](docs/demo-mode.md)
- [User help](https://argussun.com/help) · [API reference](https://argussun.com/api/v1/docs)
