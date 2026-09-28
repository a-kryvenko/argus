# LEO drag assessment

## Dashboard panel

`/dashboard/risk/leo` renders the standalone `LeoDragView` with orbit/spacecraft
inputs, 24/48-hour calculations, summary metrics, an altitude-loss chart and an
expandable hourly table. It uses the public assessment endpoint below. Access
currently follows the dashboard session; no additional permission is required.
Risk policy is deferred: the panel sends no thresholds and displays physical
estimates without assigning risk categories. Returned input parameters and UTC
source timestamps accompany each result so editing the form does not change the
meaning of the previous calculation.

## API

`POST /api/v1/public/risks/leo-drag` estimates atmospheric drag for a circular
orbit between 200 and 800 km. The API calls the internal Intelligence HTTP
service; Intelligence reads the latest published atmospheric-density release
from Prophet and calls the private `intelligence-core` backend. No collection,
forecast generation, database writes or orbit catalog lookup occurs on request.

Example (when accessing Uvicorn directly, omit the `/api/v1` proxy prefix):

```bash
curl -X POST http://localhost:8000/public/risks/leo-drag \
  -H 'Content-Type: application/json' \
  -d '{
    "altitude_km": 400,
    "inclination_deg": 51.6,
    "mass_kg": 100,
    "effective_area_m2": 1,
    "drag_coefficient": 2.2,
    "horizon_hours": 24,
    "thresholds": {
      "elevated_altitude_loss_m": 100,
      "high_altitude_loss_m": 500
    }
  }'
```

The example spacecraft parameters and thresholds are illustrative, not default
operational limits. Supply the effective area for the intended attitude and an
appropriate drag coefficient. Unknown request fields are rejected: eccentricity,
TLE and attitude profiles are not supported in this version.

The successful response uses the standard `{success, data, error}` envelope.
`data` contains:

| Field | Meaning |
| --- | --- |
| `mean_density_kg_m3` | Density averaged over orbital phase and the requested time interval |
| `mean_drag_accel_m_s2` | Mean magnitude of acceleration due to drag |
| `delta_v_loss_m_s` | Accumulated along-track drag impulse per unit mass; not the difference between initial and final orbital speed |
| `estimated_altitude_loss_m` | Positive first-order loss of circular-orbit altitude |
| `drag_risk` | `low`, `elevated`, `high`, or `not_assessed` |
| `predictions` | Hourly orbit means and cumulative losses, including zero loss at lead 0 |
| `inputs`, `source`, `assumptions` | Echoed parameters, release ID, density driver provenance and model limitations |

Thresholds refer to the **total loss over the selected horizon**, not a daily
rate. Equality counts as a threshold crossing. Without thresholds the response
still includes physical estimates, with `drag_risk: not_assessed`. Categories
are comparisons against user limits, not calibrated probabilities. Spatial
longitude percentiles are not used as confidence intervals.

The horizon is 24 or 48 hours **from the density release issue time**, explicitly
returned as `start_time`; it does not start at request time. `computed_at` and
`end_time` are also returned. Drivers are held at observed values: this is a
persistence scenario, not a prediction of future storms. Source observation and
DTC timestamps remain visible even when older than the release.

The service assumes a spherical Earth, a circular orbit, a co-rotating atmosphere
without winds, constant area and coefficient, and longitude-averaged density.
There is no station keeping or orbit propagation. A fixed-altitude approximation
is rejected above 1000 m estimated loss; this is a conservative scope restriction,
not a validated error bound. Model details and numerical tests live in the private
backend repository.

Errors:

- `422`: invalid parameters/thresholds, orbit outside the supplied grid, or decay
  above the fixed-orbit approximation's limit.
- `503`: missing configuration/backend, unreachable service, missing, malformed,
  incomplete, future-dated or stale density release. Freshness uses
  `models.atmospheric_density.max_age_hours` (default 6 hours).

## Running

Both Compose files include `intelligence-api`, using the same image as the
existing Intelligence worker. `intelligence serve` starts the internal HTTP
service on port 8000. API needs `INTELLIGENCE_URL`; API and the internal service
share `INTELLIGENCE_SERVICE_TOKEN`. Compose defaults that token to the existing
`FORECASTS_SERVICE_TOKEN` for compatibility; set a separate token to isolate the
two HTTP boundaries. Intelligence also needs `FORECASTS_URL`,
`FORECASTS_SERVICE_TOKEN` and access to project configs. The impact HTTP service
needs no database credentials.

After updating the private backend and public application code locally:

```bash
./argus compose up -d --build intelligence-api api
```

`/health/ready` checks local configuration and backend availability; density
readiness is checked per assessment. The existing worker still records forecast
integration results; it does not precompute satellite assessments.

Production requires publishing the corresponding private `intelligence-core`
commit before building the Intelligence image. Public source alone does not
contain the numerical implementation.
