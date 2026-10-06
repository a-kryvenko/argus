# Training and artifact delivery

This diagram documents the supported **public AIA/Ridge solar-wind training
pipeline**, not a claim that every model uses the same training steps. Arrows
show data or processing progression. Commands and provenance checks are described
in the [training README](../../../scripts/training/aia_wind/README.md).

```mermaid
flowchart TD
    archive["Local historical inputs<br/>OMNI, AIA193 originals and frozen DLinear"] --> features["features<br/>Extract AIA features from local archive"]
    features --> prepare["prepare<br/>Build aligned yearly frames and observed targets"]
    archive -->|Targets and baseline| prepare
    plan["plan.json<br/>Time windows, validation quarters and horizon buckets"] --> prepare
    prepare --> fit["fit<br/>Temporal validation, scale selection and Ridge refit"]
    plan --> fit
    fit --> selected[("Saved selected model<br/>Hashes and provenance")]
    selected --> evaluate["evaluate<br/>Historical 2025 assessment<br/>Repeatedly explored; not an independent test"]
    prepare -->|Historical targets| evaluate
    selected --> export["export_model.py export<br/>Package model and 2024 residual uncertainty"]
    prepare -->|2024 validation residuals| export
    export --> bundle[("Service artifact<br/>joblib bundle + SHA256")]
    bundle --> audit["Frozen artifact evaluation<br/>Check prediction parity; historical metrics"]
    evaluate -->|Reference predictions| audit
    audit --> metrics[("Static evaluation files<br/>data/metrics/plasma")]
    bundle --> deploy["Explicit deployment<br/>Artifact + configured model registry"]
    deploy --> prophet["Prophet<br/>Load artifact for forecast generation"]
```

`features`, `prepare`, `fit` and `evaluate` are explicit commands; evaluation does
not silently retrain a stale model. The service exporter packages the selected
model and validation residual uncertainty, checks prediction parity and writes
historical metrics. Deployment of the artifact is a separate action.

The 2025 period has been repeatedly explored and is not an independent test.
Optional model evaluation and MLflow logging in local notebooks are separate
from this public command pipeline. Private training implementations remain in
their respective private repositories.

See [operational verification](verification.md) for evaluation of published
releases and the [architecture overview](../README.md).
