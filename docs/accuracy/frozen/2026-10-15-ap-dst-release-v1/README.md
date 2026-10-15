# Ap/Dst production models

Registry: `argus-ap-revalidated-2026-v1`, `argus-dst-revalidated-2026-v1`. Both produce q10/q50/q90 through 120 h; ap targets three-hour intervals, Dst hourly values.

Train 2011–2024; early stopping Jan–Sep 2025; residual quantile calibration Oct–Dec 2025. Frozen estimators and calibration are unchanged. [Methodology](../../METHODOLOGY.md).

Released by explicit user decision accepting accuracy reductions for consistent training and validation. Production inference exactly reproduces 1,329,960 saved calibration/test rows; full snapshot checks cover 120 h per model.

Release evidence: `data/revalidation/frozen/2026-10-15-ap-dst-release-v1/`. Rollback: restore its `models_registry.before.yaml` and restart Prophet/API. Previous weights and metrics are retained.
