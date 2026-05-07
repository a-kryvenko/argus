#!/usr/bin/env python3
"""
Build Argus Sunwatch v1 speed-forecast training dataset.

One output row = (issue_time, valid_time, lead_hours).

Inputs
------
1) HUXt ensemble output with columns:
   issue_time, valid_time, lead_hours, member_id, v_huxt

2) L1/OMNI/DSCOVR observations with columns:
   time, v_obs, n_obs, bz_obs, bt_obs, kp

Output
------
CSV with fixed v1 feature list + targets:
   ensemble statistics
   forecast geometry
   persistence features
   plasma/context features
   error-memory features
   target_v and event labels

Example
-------
python build_speed_dataset.py \
  --huxt-input ../assets/huxt_ensemble.csv \
  --obs-input ../assets/omni_hourly.csv \
  --output ../assets/speed_dataset_v1.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SOLAR_ROTATION_DAYS = 27.2753
SOLAR_ROTATION_HOURS = SOLAR_ROTATION_DAYS * 24.0

REQUIRED_HUXT_COLUMNS = {
    "issue_time",
    "valid_time",
    "lead_hours",
    "member_id",
    "v_huxt",
}

REQUIRED_OBS_COLUMNS = {
    "time",
    "v_obs",
    "n_obs",
    "bz_obs",
    "bt_obs",
    "kp",
}

OUTPUT_COLUMNS = [
    # keys
    "issue_time",
    "valid_time",
    "lead_hours",
    # HUXt ensemble
    "ens_mean",
    "ens_std",
    "ens_min",
    "ens_max",
    "ens_p10",
    "ens_p50",
    "ens_p90",
    "ens_spread",
    "ens_skew_proxy",
    # forecast geometry
    "lead_norm",
    "valid_sin_27d",
    "valid_cos_27d",
    # persistence
    "v_persist_1h",
    "v_persist_6h",
    "v_persist_24h",
    "v_persist_27d",
    # persistence deltas
    "delta_v_1h_6h",
    "delta_v_1h_24h",
    "delta_v_24h_27d",
    # current plasma/context
    "n_obs",
    "bz_obs",
    "bt_obs",
    "kp",
    "abs_bz",
    "southward_bz",
    # error memory
    "recent_model_bias_24h",
    "recent_model_bias_72h",
    "recent_model_abs_error_24h",
    "recent_model_abs_error_72h",
    # target
    "target_v",
    "target_high_speed_500",
    "target_high_speed_600",
    "target_high_speed_700",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")

    raise ValueError(f"Unsupported input format: {path}. Use CSV, Parquet, JSON, or JSONL.")


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalize_time_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], utc=True).dt.floor("h")
    return df


def clean_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_ensemble_features(huxt: pd.DataFrame) -> pd.DataFrame:
    huxt = huxt.copy()
    huxt = normalize_time_column(huxt, "issue_time")
    huxt = normalize_time_column(huxt, "valid_time")
    huxt = clean_numeric(huxt, ["lead_hours", "v_huxt"])
    huxt = huxt.dropna(subset=["issue_time", "valid_time", "lead_hours", "v_huxt"])

    grouped = huxt.groupby(["issue_time", "valid_time", "lead_hours"], as_index=False)

    ens = grouped["v_huxt"].agg(
        ens_mean="mean",
        ens_std="std",
        ens_min="min",
        ens_max="max",
        ens_p10=lambda x: float(np.nanpercentile(x, 10)),
        ens_p50=lambda x: float(np.nanpercentile(x, 50)),
        ens_p90=lambda x: float(np.nanpercentile(x, 90)),
    )

    # std is NaN for a single-member ensemble; keep the feature usable.
    ens["ens_std"] = ens["ens_std"].fillna(0.0)
    ens["ens_spread"] = ens["ens_p90"] - ens["ens_p10"]
    ens["ens_skew_proxy"] = ens["ens_mean"] - ens["ens_p50"]

    return ens


def prepare_observations(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    obs = normalize_time_column(obs, "time")
    obs = clean_numeric(obs, ["v_obs", "n_obs", "bz_obs", "bt_obs", "kp"])
    obs = obs.dropna(subset=["time", "v_obs"])
    obs = obs.sort_values("time")

    # If several records exist for the same hour, use the hourly mean.
    obs = (
        obs.groupby("time", as_index=False)
        .agg(
            v_obs=("v_obs", "mean"),
            n_obs=("n_obs", "mean"),
            bz_obs=("bz_obs", "mean"),
            bt_obs=("bt_obs", "mean"),
            kp=("kp", "mean"),
        )
        .sort_values("time")
    )

    return obs


def add_current_context(df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    current = obs.rename(columns={"time": "issue_time"})[
        ["issue_time", "n_obs", "bz_obs", "bt_obs", "kp"]
    ]
    out = df.merge(current, on="issue_time", how="left")
    out["abs_bz"] = out["bz_obs"].abs()
    out["southward_bz"] = np.maximum(0.0, -out["bz_obs"])
    return out


def add_targets(df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    target = obs.rename(columns={"time": "valid_time", "v_obs": "target_v"})[
        ["valid_time", "target_v"]
    ]
    out = df.merge(target, on="valid_time", how="left")
    out["target_high_speed_500"] = (out["target_v"] >= 500.0).astype("Int64")
    out["target_high_speed_600"] = (out["target_v"] >= 600.0).astype("Int64")
    out["target_high_speed_700"] = (out["target_v"] >= 700.0).astype("Int64")
    return out


def add_persistence(df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    v_by_time = obs[["time", "v_obs"]].copy()

    offsets = {
        "v_persist_1h": pd.Timedelta(hours=1),
        "v_persist_6h": pd.Timedelta(hours=6),
        "v_persist_24h": pd.Timedelta(hours=24),
        "v_persist_27d": pd.Timedelta(days=27),
    }

    for feature_name, offset in offsets.items():
        lookup = v_by_time.rename(columns={"time": f"{feature_name}_time", "v_obs": feature_name})
        out[f"{feature_name}_time"] = out["issue_time"] - offset
        out = out.merge(
            lookup,
            on=f"{feature_name}_time",
            how="left",
        )
        out = out.drop(columns=[f"{feature_name}_time"])

    out["delta_v_1h_6h"] = out["v_persist_1h"] - out["v_persist_6h"]
    out["delta_v_1h_24h"] = out["v_persist_1h"] - out["v_persist_24h"]
    out["delta_v_24h_27d"] = out["v_persist_24h"] - out["v_persist_27d"]

    return out


def add_temporal_features(df: pd.DataFrame, max_lead_hours: int) -> pd.DataFrame:
    out = df.copy()
    valid_hours = out["valid_time"].astype("int64") / 3_600_000_000_000
    phase = 2.0 * math.pi * valid_hours / SOLAR_ROTATION_HOURS

    out["lead_norm"] = out["lead_hours"] / float(max_lead_hours)
    out["valid_sin_27d"] = np.sin(phase)
    out["valid_cos_27d"] = np.cos(phase)

    return out


def compute_recent_error_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute recent error-memory features without leakage.

    At a given issue_time, only forecasts with valid_time <= issue_time are allowed.
    Error is defined as ens_mean - observed target_v.
    """
    out = df.copy()
    history = out.dropna(subset=["target_v", "ens_mean"])[
        ["valid_time", "ens_mean", "target_v"]
    ].copy()
    history["error"] = history["ens_mean"] - history["target_v"]
    history["abs_error"] = history["error"].abs()
    history = history.sort_values("valid_time")

    issue_times = pd.DataFrame({"issue_time": sorted(out["issue_time"].dropna().unique())})

    for hours in (24, 72):
        bias_values: list[float] = []
        abs_values: list[float] = []
        window = pd.Timedelta(hours=hours)

        # This is intentionally explicit and easy to audit. Dataset size is expected to be moderate.
        for issue_time in issue_times["issue_time"]:
            mask = (history["valid_time"] <= issue_time) & (history["valid_time"] > issue_time - window)
            h = history.loc[mask]
            bias_values.append(float(h["error"].mean()) if len(h) else np.nan)
            abs_values.append(float(h["abs_error"].mean()) if len(h) else np.nan)

        issue_times[f"recent_model_bias_{hours}h"] = bias_values
        issue_times[f"recent_model_abs_error_{hours}h"] = abs_values

    out = out.merge(issue_times, on="issue_time", how="left")
    return out


def build_dataset(
    huxt_input: str | Path,
    obs_input: str | Path,
    output: str | Path,
    max_lead_hours: int = 96,
    min_lead_hours: int = 1,
    drop_missing: bool = True,
) -> pd.DataFrame:
    huxt = read_table(huxt_input)
    obs = read_table(obs_input)

    require_columns(huxt, REQUIRED_HUXT_COLUMNS, "HUXt input")
    require_columns(obs, REQUIRED_OBS_COLUMNS, "Observation input")

    obs = prepare_observations(obs)
    df = build_ensemble_features(huxt)

    df = df[(df["lead_hours"] >= min_lead_hours) & (df["lead_hours"] <= max_lead_hours)]
    df = add_temporal_features(df, max_lead_hours=max_lead_hours)
    df = add_persistence(df, obs)
    df = add_current_context(df, obs)
    df = add_targets(df, obs)
    df = compute_recent_error_memory(df)

    # Stable order.
    df = df.sort_values(["issue_time", "lead_hours", "valid_time"]).reset_index(drop=True)
    df = df[OUTPUT_COLUMNS]

    if drop_missing:
        before = len(df)
        df = df.dropna(subset=OUTPUT_COLUMNS).reset_index(drop=True)
        dropped = before - len(df)
        print(f"Dropped rows with missing required values: {dropped}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    if len(df):
        print(f"Issue time range: {df['issue_time'].min()} → {df['issue_time'].max()}")
        print(f"Valid time range: {df['valid_time'].min()} → {df['valid_time'].max()}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Argus Sunwatch v1 solar-wind speed dataset CSV.")
    parser.add_argument("--huxt-input", required=True, help="Path to HUXt ensemble CSV/Parquet/JSONL.")
    parser.add_argument("--obs-input", required=True, help="Path to OMNI/DSCOVR observations CSV/Parquet/JSONL.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--max-lead-hours", type=int, default=96)
    parser.add_argument("--min-lead-hours", type=int, default=1)
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep rows with NaN values instead of dropping incomplete training rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        huxt_input=args.huxt_input,
        obs_input=args.obs_input,
        output=args.output,
        max_lead_hours=args.max_lead_hours,
        min_lead_hours=args.min_lead_hours,
        drop_missing=not args.keep_missing,
    )


if __name__ == "__main__":
    main()
