# Forecast publication

Arrows show data or processing progression. Products publish independently.
A failed attempt retains the previous release when one exists; it does not
create a synthetic fallback release. Calculation processes receive saved inputs;
the coordinator owns database writes. Public reads never trigger generation.

```mermaid
flowchart TD
    observations["Clio HTTP<br/>Observations and original files"] --> prepare["Prophet<br/>Prepare features and calibrate supplied inputs"]
    prepare --> snapshot[("Prophet DB<br/>Saved input snapshot")]
    snapshot --> calculate["Per-product calculation processes<br/>Saved inputs; no database writes"]
    models["Configured model artifacts<br/>forecast / forecast-core"] --> calculate
    calculate --> save["Prophet coordinator<br/>Save serialized results"]
    save --> complete{"Product complete?"}
    complete -->|Yes| publish["Publish this product release"]
    complete -->|No| previous["Keep previous published release<br/>If one exists"]
    previous --> retry["Retry failed products<br/>Within current schedule slot"]
    publish --> http["Prophet HTTP<br/>Stored releases"]
    previous --> http
    http --> api["Public API"]
    api --> clients["Web UI and external clients"]
```

See [Prophet](../../../apps/prophet/README.md) and the
[architecture overview](../README.md).
