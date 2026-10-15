# CME correction in solar-wind speed

The main speed forecast includes the existing CME scenario increment with a triangular weight: zero at predicted early arrival, one at nominal arrival, zero at late arrival +12 h. No window expansion. Overlapping increments use their maximum; unavailable CME inputs retain the background forecast. The existing CME horizon is 96 h.

Speed quantiles shift by the same increment; exceedance probabilities use the translated background residual distribution. CME timing/amplitude uncertainty remains uncalibrated. Background q50 is retained in each artifact.

Retrospective 2026 evaluation: 16 linked arrivals, one first issue per physical event, equal event weights. Inside predicted windows, speed MAE 72.31→72.26 km/s; peak MAE 115.73→71.30 km/s. Paired intervals include zero. Scientific historical baseline, not native NRT validation. Existing CME release uses 2025H1 calibration / H2 selection; its upstream split differs from the current methodology.

Evaluation: `data/revalidation/frozen/2026-10-15-cme-event-window-v3/`. Release evidence: `data/revalidation/internal-reports/releases/2026-10-15-cme-window-v1/`. Rollback: set `models.solar_event_speed.integration.enabled: false` in `configs/models_registry.yaml`, then restart Prophet. Weights are unchanged.
