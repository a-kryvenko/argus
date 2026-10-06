# Observation collection and preparation

Arrows show data movement. Clio owns source history and original-file receipts.
Prophet derives model features from downloaded originals and owns a disposable
derivative cache. Public reads do not trigger collection or preparation.

```mermaid
flowchart TD
    providers["External observation providers"] --> collect["Clio worker<br/>Live collection and historical backfill"]
    collect -->|Parsed measurements and receipts| raw[("Clio DB<br/>Measurements and archive receipts")]
    collect -->|Original files| files[("Clio original file archive<br/>AIA / HMI / GONG")]
    raw --> prepare["Clio worker<br/>Aggregation and normalization"]
    prepare --> prepared[("Clio DB<br/>Aggregates and normalized observations")]
    raw --> http["Clio HTTP<br/>Stored observations and original-file API"]
    prepared --> http
    files -->|Metadata and checksum-verified originals| http
    http -->|Forecast inputs| prophet["Prophet<br/>Input preparation and derivative cache"]
    http -->|Observation contracts| api["Public API"]
```

See [Clio](../../../apps/clio/README.md),
[Prophet](../../../apps/prophet/README.md) and the
[architecture overview](../README.md).
