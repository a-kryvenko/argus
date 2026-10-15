# IMF production models

Registry: `argus-bt-revalidated-2026-v2` (1–24 h), `argus-bs-revalidated-2026-v2` (1–48 h). Both predict exceedance probabilities at 5/10/15 nT from native hourly solar-wind history; GONG is not required.

Train 2011–2024; early stopping Jan–Sep 2025; Platt calibration Oct–Dec 2025. Frozen weights installed unchanged. [Methodology](../../METHODOLOGY.md).

Released by explicit user decision after comparison; original accuracy gates remain failed. [Bt/Bs evaluation](../2026-10-15-imf-v2/README.md) · [Bs/GONG comparison](../2026-10-15-imf-gong-v1/README.md).

Release evidence and previous configuration: `data/revalidation/frozen/2026-10-15-imf-release-v1/`. Rollback: restore its `models_registry.before.yaml` and `project.before.yaml`, then restart Prophet and API. Previous artifacts and metrics are retained.
