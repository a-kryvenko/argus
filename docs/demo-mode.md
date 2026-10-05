# Historical demo

Public route: `/demo`. Replay runs hourly from T−96 to T0, with forecasts and
optional actual future values. T0 is the **G4 onset on 19 January 2026 at
19:38 UTC** ([NOAA SWPC](https://swpc-drupal.woc.noaa.gov/news/g4-severe-geomagnetic-storm-levels-reached-19-jan-2026)).

## Build or update

Run from the installation directory in dev or production:

```bash
./argus demo                   # Build using cached archives
./argus demo --refresh-data    # Download archives again and rebuild
./argus demo january-2026      # Select a scenario
```

Requires installed Prophet models and Docker Compose; archive downloads need
HTTPS access. The command prepares inputs, generates 97 forecast issues per
product and atomically publishes `data/demo/current.json`. Failures preserve
the previous bundle. Reload the page to see the new version.

Scenarios: `configs/demo/{name}.json`. After changing models, update their hashes
and training cutoff evidence in the scenario, then rerun the command.
There is no automatic rebuild schedule. Demo forecasts are generated separately
from operational forecasts and do not train models or affect operational scores.

## Available forecasts

| Product | Output | Horizon |
| --- | --- | --- |
| Solar wind speed | v quantiles and threshold probabilities | 96 h |
| Proton density | n quantiles | 96 h |
| Geomagnetic activity | Kp probabilities, Ap quantiles | 48 h |
| Dst | Dst quantiles | 48 h |
| Magnetic field | Bt threshold probabilities | 24 h |
| Solar indices | F10.7 quantiles | 48 h |

## Data and limitations

- Inputs and actual outcomes come from revised NASA OMNI2 hourly archives;
  original real-time delivery delays are not reproduced.
- Each forecast uses only completed historical intervals. Training and
  calibration must end before T−96.
- Legacy Kp/Ap, Dst and F10.7 training cutoffs rely on saved recipes and MLflow
  records; their model files do not contain training dates.
- Speed uses the DLinear fallback without AIA. Bt uses OMNI features matching
  its training data and provides probabilities, not magnitude quantiles.
- Bs requires historical GONG inputs and is unavailable. S10/M10/Y10 have no
  installed models.
