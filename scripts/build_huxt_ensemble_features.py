#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def load_huxt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "issue_time",
        "valid_time",
        "lead_hours",
        "member_id",
        "v_huxt",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing HUXt columns: {missing}")

    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["lead_hours"] = df["lead_hours"].astype(int)
    df["v_huxt"] = pd.to_numeric(df["v_huxt"], errors="coerce")

    df = df.dropna(subset=["v_huxt"])

    return df


def build_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["issue_time", "valid_time", "lead_hours"]

    # One row per forecast target, columns = member_id
    wide = (
        df.pivot_table(
            index=group_cols,
            columns="member_id",
            values="v_huxt",
            aggfunc="mean",
        )
        .sort_index()
    )

    values = wide.to_numpy(dtype=np.float32)

    out = wide.reset_index()[group_cols].copy()

    out["ens_mean"] = np.nanmean(values, axis=1)
    out["ens_std"] = np.nanstd(values, axis=1)
    out["ens_min"] = np.nanmin(values, axis=1)
    out["ens_max"] = np.nanmax(values, axis=1)
    out["ens_p10"] = np.nanpercentile(values, 10, axis=1)
    out["ens_p50"] = np.nanpercentile(values, 50, axis=1)
    out["ens_p90"] = np.nanpercentile(values, 90, axis=1)
    out["ens_members"] = np.sum(~np.isnan(values), axis=1)

    out["ens_spread"] = out["ens_p90"] - out["ens_p10"]
    out["ens_skew_proxy"] = out["ens_mean"] - out["ens_p50"]

    return out.sort_values(["issue_time", "lead_hours"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HUXt ensemble features from per-member HUXt forecasts."
    )

    parser.add_argument(
        "--huxt",
        required=True,
        type=Path,
        help="CSV with issue_time, valid_time, lead_hours, member_id, v_huxt.",
    )

    parser.add_argument(
        "--output",
        default=Path("data/processed/huxt_ensemble_features.csv"),
        type=Path,
    )

    args = parser.parse_args()

    huxt = load_huxt(args.huxt)
    features = build_ensemble_features(huxt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    print(f"saved {len(features)} rows to {args.output}")
    print(features.head())


if __name__ == "__main__":
    main()