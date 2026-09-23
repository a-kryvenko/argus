"""Observed hourly targets and the selected AIA feature schema."""
from pathlib import Path
import numpy as np
import pandas as pd
from aia_features import attach_features

def load_year(year, oof_dir, omni_dir, features, *, leads, stride_hours=6, max_age_hours=12):
    """Load observed labels only; exclude targets crossing the calendar boundary."""
    if stride_hours < 1 or not leads or min(leads) < 1 or max(leads) > 120:
        raise ValueError("Invalid horizons or issue stride")
    path = Path(oof_dir)/f"year={year}.parquet"
    frame = pd.read_parquet(path, filters=[("lead_hours", "in", list(leads))])
    for column in ["issue_time", "valid_time", "model_train_end"]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame.duplicated(["issue_time", "lead_hours"]).any():
        raise ValueError("Duplicate DLinear issue/horizon rows; choose one strategy")
    if not (frame.model_train_end <= frame.issue_time).all():
        raise ValueError("DLinear forecasts are not chronological OOF")
    if not (frame.valid_time == frame.issue_time+pd.to_timedelta(frame.lead_hours, unit="h")).all():
        raise ValueError("Invalid DLinear valid_time")
    if not (frame.issue_time.dt.year == year).all() or frame.strategy.nunique() != 1:
        raise ValueError("OOF file contains an unexpected year/strategy")
    # Timestamp.hour preserves UTC six-hour sampling across year boundaries.
    elapsed = (frame.issue_time-pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(hours=1)
    frame = frame[(elapsed % stride_hours == 0) & (frame.valid_time.dt.year == year)].copy()
    observations = pd.read_parquet(Path(omni_dir)/f"omni_{year}.parquet", columns=["issue_time", "v"])
    observations["valid_time"] = pd.to_datetime(observations.issue_time, utc=True)
    observations = observations[["valid_time", "v"]].rename(columns={"v": "target_v"})
    if observations.valid_time.duplicated().any():
        raise ValueError("Duplicate OMNI timestamps")
    observations.target_v = observations.target_v.replace([9999., np.inf, -np.inf], np.nan)
    frame = frame.merge(observations, on="valid_time", how="left", validate="many_to_one")
    aligned = attach_features(frame.issue_time, features, max_age_hours=max_age_hours)
    frame = frame.merge(aligned, on="issue_time", how="left", validate="many_to_one")
    phase = 2*np.pi*(frame.issue_time.dt.dayofyear-1)/365.25
    frame["calendar_sin"] = np.sin(phase)
    frame["calendar_cos"] = np.cos(phase)
    if not np.isfinite(frame.dlinear_v).all():
        raise ValueError("Non-finite DLinear forecasts")
    # Retain missing target rows for coverage audit and exported forecasts.
    return frame

def feature_columns(features):
    control = ["dlinear_v", "lead_hours", "calendar_sin", "calendar_cos",
               "aia_age_hours", "aia_valid_fraction", "aia_b0_deg"]
    current = [c for c in features if c.startswith("aia_area_")]
    changes = [c for c in features if c.startswith(("aia_delta_", "aia_overlap_"))
               or c.endswith("_separation_h")]
    if not current or not changes:
        raise ValueError("Missing daily AIA area/change features")
    return control + current + changes
