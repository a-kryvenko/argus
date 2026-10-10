"""Experimental ap-quantile -> continuous Kp conversion and DTM driver adapter.

python scripts/training/thermosphere/ap_forecast.py INPUT.csv OUTPUT.csv

GFZ table: https://www.gfz.de/en/section/geomagnetism/data-products-services/kp-index/related-indices
Linear interpolation between tabulated ap values follows the DTM input convention.
Hourly forecasts retain their timestamps; these are not official 3-hour Kp bins.
q10/q90 trajectories are sensitivity scenarios, not density confidence bounds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models import AP, KP, causal_interpolate, dtm_snapshot


def convert_ap_forecasts(frame):
    result = frame.copy()
    required = {"issue_time", "valid_time", "ap_q10", "ap_q50", "ap_q90"}
    if not required.issubset(result):
        raise ValueError(f"Missing forecast columns: {sorted(required-set(result))}")
    for column in ("issue_time", "valid_time"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if (result[["issue_time", "valid_time"]].isna().any().any()
            or not result.valid_time.gt(result.issue_time).all()
            or result.duplicated(["issue_time", "valid_time"]).any()):
        raise ValueError("Forecast times must be unique and strictly after issue_time")
    values = result[["ap_q10", "ap_q50", "ap_q90"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ValueError("ap quantiles must be finite and ordered")
    for i, quantile in enumerate(("q10", "q50", "q90")):
        result[f"ap_{quantile}_clipped"] = (values[:, i] < 0) | (values[:, i] > 400)
        result[f"kp_{quantile}"] = np.interp(np.clip(values[:, i], 0, 400), AP, KP)
    result["kp_method"] = "gfz_ap_piecewise_linear_clipped_0_400"
    return result.sort_values(["issue_time", "valid_time"]).reset_index(drop=True)


def forecast_dtm_snapshot(measurements, forecasts, issue_time, valid_time, quantile="q50", *, baseline=None):
    """Keep solar drivers fixed; vary only ap-derived Kp for an isolated experiment.

    Uses exactly one forecast issuance and observations available at that issuance.
    Never extrapolates beyond forecast coverage. Interpolates ap before converting
    to Kp, applies the three-hour lag and a trailing 24-hour mean of hourly Kp.
    """
    if quantile not in ("q10", "q50", "q90", "q95"):
        raise ValueError("Unknown quantile")
    issue, valid = pd.Timestamp(issue_time), pd.Timestamp(valid_time)
    if issue.tzinfo is None or valid.tzinfo is None or valid < issue:
        raise ValueError("Require timezone-aware valid_time >= issue_time")
    baseline = dtm_snapshot(measurements, issue) if baseline is None else baseline
    if valid == issue:
        return baseline
    if {"ap_q10", "ap_q50", "ap_q90"}.issubset(forecasts):
        forecasts = convert_ap_forecasts(forecasts)
    else:
        # A median-only experimental model must not invent q10/q90 intervals.
        forecasts = forecasts.copy()
        column = f"ap_{quantile}"
        if column not in forecasts:
            raise ValueError(f"Missing {column} forecast")
        for name in ("issue_time", "valid_time"):
            forecasts[name] = pd.to_datetime(forecasts[name], utc=True, errors="raise")
        if (forecasts[["issue_time", "valid_time"]].isna().any().any()
                or forecasts.duplicated(["issue_time", "valid_time"]).any()
                or not forecasts.valid_time.gt(forecasts.issue_time).all()
                or not np.isfinite(forecasts[column].to_numpy(dtype=float)).all()):
            raise ValueError("Invalid single-quantile forecast")
    selected = forecasts.loc[forecasts.issue_time.eq(issue)]
    if selected.empty or selected.valid_time.max() < valid:
        raise ValueError("Forecast issuance or horizon unavailable")
    known = measurements.loc[measurements.metric.eq("ap")
        & measurements.available_at.le(issue) & measurements.observed_at.le(issue)
        & measurements.observed_at.ge(issue-pd.Timedelta(hours=30))]
    past = known.set_index("observed_at").value.sort_index()
    if (past.empty or not past.between(0, 400).all() or past.index.has_duplicates
            or issue-past.index[-1] > pd.Timedelta(hours=6)
            or (past.index.to_series().diff().dropna() > pd.Timedelta(hours=3)).any()):
        raise ValueError("Invalid observed ap history")
    # Bridge the unfinished bin with persistence at issue_time, never future truth.
    past.loc[issue] = past.iloc[-1]
    future = selected.set_index("valid_time")[f"ap_{quantile}"].clip(0, 400)
    series = pd.concat([past, future]).sort_index()
    # Closed ap is labelled at interval START: its age is 3–6 hours, so the
    # bridge to issue_time can exceed 3h without a missing observed bin.
    future_clock = pd.DatetimeIndex([issue, *future.index])
    if series.index.has_duplicates or (future_clock.to_series().diff().dropna() > pd.Timedelta(hours=3)).any():
        raise ValueError("Gap in ap observation/forecast trajectory")
    times = pd.DatetimeIndex([valid-pd.Timedelta(hours=3),
        *pd.date_range(valid-pd.Timedelta(hours=23), valid, freq="h")])
    ap = causal_interpolate(series, times, pd.Timedelta(0))
    kp = np.interp(ap, AP, KP)
    return {**baseline, "kp": float(kp[0]), "kp_mean": float(kp[1:].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = convert_ap_forecasts(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Converted {len(result)} rows; clipped quantiles: "
          f"{result.filter(like='_clipped').sum().to_dict()}")


if __name__ == "__main__":
    main()
