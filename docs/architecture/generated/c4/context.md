# Context

Generated from [workspace.dsl](../../workspace.dsl). Do not edit by hand.

Arrows show requests or storage access, not response data.

```mermaid
graph LR
  linkStyle default fill:#ffffff

  subgraph diagram ["System Context View: Argus Sunwatch"]
    style diagram fill:#ffffff,stroke:#ffffff

    1["User<br/>[Person]<br/>Explores space weather and<br />forecasts"]
    style 1 fill:#eff6ff,stroke:#a7acb2,color:#172554
    2["External API client<br/>[Software System]<br/>Consumes public contracts"]
    style 2 fill:#eff6ff,stroke:#a7acb2,color:#172554
    3["Observation providers<br/>[Software System]<br/>SWPC, OMNI, JSOC, GONG and<br />other sources"]
    style 3 fill:#eff6ff,stroke:#a7acb2,color:#172554
    4["Argus Sunwatch<br/>[Software System]<br/>Collects observations,<br />publishes forecasts and<br />assesses impacts"]
    style 4 fill:#eff6ff,stroke:#a7acb2,color:#172554

    1-. "Explores observations and<br />forecasts<br/>[HTTPS]" .->4
    2-. "Reads public contracts<br/>[HTTPS]" .->4
    4-. "Fetches observations and<br />original files<br/>[HTTP]" .->3

  end
```

[Architecture overview](../../README.md)
