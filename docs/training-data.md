# Historical training data preparation

Run `notebooks/0_omni_raw_data_fetching.ipynb`, then `1_omni_data_normalize.ipynb`,
`2_data_processing.ipynb`, and `3_build_training_dataset.ipynb` for each split
(`2010_2024`, `2025`). Set matching dataset names in each notebook.

The OMNI fetch notebook defaults to keeping cached downloads. Set `REFRESH=True`
to recover fractional Kp and the final 23 hours omitted by the former loader.
Cleaning old raw CSVs removes remaining fill codes, but cannot recover truncated
Kp or observations already discarded by the former parser.

- `data/clean/omni_*.csv` contains unfilled observations on a sorted hourly grid.
- `data/processed/observations_*.csv` contains features filled from the past for
  at most three hours, original-observation flags, and trailing statistics.
- `data/training/<split>` joins these features to unfilled observations by exact
  target timestamp across the entire split. Missing targets remain NaN, including
  threshold labels. Training notebooks filter only their selected target columns.
- Evaluation filters missing targets and missing persistence inputs so model and
  persistence use the same origins. Other feature NaNs remain available to LightGBM.
- `_preparation.json` records the target date range, row counts and missing targets.
  A completed rebuild replaces generated shards and removes stale numbered shards.
  Do not run readers or concurrent writers during replacement.

Regenerate GONG features from original FITS using `0_gong_fits_unpack.ipynb` and
`1_gong_extract_features.ipynb`: previously mean-filled CSVs cannot be cleaned
retrospectively. FITS filenames preserve minutes, hourly bins are right-labelled,
and gaps remain missing until causal feature preparation. Existing renamed FITS
with lost minute information require recovery from original source files.

These are retrospective observation-time benchmarks. They do not yet simulate
source publication delays: hourly OMNI averages, three-hour Kp/Ap and daily F10.7
are not necessarily available operationally at the recorded hour. Matching the
live inputs' availability times remains a separate requirement for operational
validation. Feature windows start at the beginning of each split.

Implementation: `forecast_core.data_pipelines.training_data`. Rebuilding data
changes both labels and features; retrain models and regenerate evaluation metrics
before using the new results. Existing models and metric files are not refreshed
by these notebooks.
