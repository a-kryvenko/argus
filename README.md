# Argus Sunwatch

**Solar Activity Impact Forecasting & Decision Intelligence**


## Overview

Argus Sunwatch is a software system designed to analyze solar activity data and provide **risk-oriented insights on potential impacts to terrestrial infrastructure**, including electrical power systems.


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
    1. Live data sources:
        - [Deep Space Climate Observatory](https://epic.gsfc.nasa.gov/) - provide monitoring of Bx, By, Bz, V (Solar Wind), N (Density), T (Plasma temperature) in L1 Lagrange point. Data accessed wia [NOAA](https://services.swpc.noaa.gov/json/)
        - [Solar Dynamics Observatory](https://data.nasa.gov/dataset/solar-dynamics-observatory) - provide solar observations. Data are loaded wia 
            [jsoc](https://jsoc1.stanford.edu/data/) (AIA, HMI)
1. Processing Layer
    - Normalize data
    - combine sensors data with solar observations
1. Forecasting Layer
    - Heuristic models (initial phase)
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
  "timestamp": "2026-03-30T12:00:00Z",
  "solar_activity_level": "Elevated",
  "risk_score": "Medium",
  "recommended_action": "Monitor grid stability and prepare mitigation procedures"
}
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
- Satellite: input orbit altitude + inclination -> proton upset %.
- GPS: simple TEC formula.
- Aviation: dose rate at FL350 or input route waypoints -> dose rate.

---

## Roadmap

- Expand real-time data integrations
- Improve forecasting models
- Enhance visualization dashboard
- Introduce anomaly detection
- Refine impact intelligence models

---

## License

Forecasting components are publicly available under a **non-commercial license**.
Commercial usage requires explicit permission from the project owner.

---

## Author

Andriy Kryvenko


Software Systems Architect | Cybersecurity & Infrastructure Specialist

---

> Note: The commercial impact intelligence module (GIC risk assessment for power grids, aviation, satelites) is a proprietary closed-source component and is not included in this repository.