# Historical demo

`/demo`: hourly replay T−96…T+72 around the
[G4 onset, January 19, 2026, 19:38 UTC](https://swpc-drupal.woc.noaa.gov/news/g4-severe-geomagnetic-storm-levels-reached-19-jan-2026).
`/demo/forecast/solar-events`: archived ENLIL arrivals and CME speed scenarios.
`/demo/leo`: JB2008/DTM2020 with fixed or forecast geomagnetic drivers, +1…+48 h.

```bash
./argus demo
./argus demo --refresh-data   # Refresh OMNI only
```

Requires the research checkout, `notebooks/.venv`, both private cores, installed
models, a licensed local DTM2020 build and pinned OMNI/GOES/GRACE-C/PROSWIN caches.
Supports `january-2026`; cache overrides: `./argus demo --help`.
Update model hashes and cutoff evidence in `configs/demo/january-2026.json` after model changes.

Publishes `data/demo/current.json` after all stages pass; failures preserve the
previous dataset. Run artifacts: `data/demo/runs/`. Each release atomically updates
production from the uploaded bundle; demo rebuilding is disabled in production.

Retrospective archives use assumed availability; this previously inspected event
is not a blind holdout. Training/calibration must end before T−96.
LEO thresholds: 2×/4× the observed median over 72 h before replay; research rules,
not spacecraft safety limits.

ENLIL uses the pinned `enlil-notifications.json` in the January demo cache.
Availability follows notification publication, not recorded Argus receipts;
ambiguous multi-CME notifications are excluded. CME context remains separate from density.
CME speed correction v2 uses a triangular peak increment: zero at early arrival,
full at expected arrival, zero at late arrival +12 h; existing background rise is deducted.
