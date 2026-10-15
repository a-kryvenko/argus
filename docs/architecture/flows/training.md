# Training and artifact delivery

Training, calibration and detailed experiments live in private notebooks/backends.
Public services load frozen artifacts; they do not fit models.

```mermaid
flowchart LR
    observations["Versioned historical observations"] --> research["Private training and calibration"]
    protocol["Frozen split and validation protocol"] --> research
    research --> model["Model artifact and SHA256"]
    model --> evaluation["Frozen evaluation and inference parity"]
    evaluation --> metrics["Static accuracy results"]
    model --> deploy["Explicit model deployment"]
    deploy --> prophet["Prophet inference"]
```

[Validation protocol](../../accuracy/METHODOLOGY.md) · [Operational verification](verification.md)
