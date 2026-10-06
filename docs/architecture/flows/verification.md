# Operational verification

Verification uses raw observations as target hours finish. Missing and pending
targets remain explicit. Scores use verified pairs and are grouped by artifact,
model hash and horizon. This process does not update static historical metrics.
Arrows show data or processing progression.

```mermaid
flowchart TD
    raw["Clio HTTP<br/>Raw measurements; no interpolation"] --> targets["Prophet<br/>UTC-hour targets from observed measurements"]
    release[("Prophet DB<br/>Existing published releases")] --> match["Match forecast and observation<br/>By valid time and artifact"]
    targets --> match
    match --> status{"Target availability"}
    status -->|Future| pending["pending<br/>Target hour not finished"]
    status -->|Finished, unavailable| missing["missing<br/>No finite observation"]
    status -->|Finished, available| verified["verified<br/>Finite observation available"]
    pending --> records[("Prophet DB<br/>Pairs, status, model hash and evidence<br/>Upsert per release and artifact")]
    missing --> records
    verified --> records
    records --> cli["CLI verification report<br/>Scores computed from verified pairs"]
    records --> http["Prophet HTTP<br/>30-day summary by valid time and lead hour"]
    http --> api["Public API and Web<br/>Operational accuracy"]
```

See the [verification protocol](../../forecast-workflows.md) and the
[architecture overview](../README.md).
