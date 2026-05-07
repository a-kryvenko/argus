
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

SOLAR_ROTATION_DAYS = 27.2753
MAX_LEAD_HOURS = 96
SOLAR_ROTATION_DAYS = 27.2753

BASE_FEATURE_COLUMNS = [
    "v_obs",
    "n_obs",
    "bz_obs",
    "bt_obs",
    "kp",

    "v_persist_1h",
    "v_persist_6h",
    "v_persist_24h",
    "v_persist_27d",

    "delta_v_1h_6h",
    "delta_v_1h_24h",
    "delta_v_24h_27d",

    "abs_bz",
    "southward_bz",

    "issue_sin_27d",
    "issue_cos_27d",
]

def _build_features(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source, parse_dates=["time"])
    
    df = df.set_index("time").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    df = df.reindex(full_index)
    df.index.name = "issue_time"
    

    # Persistence features.
    df["v_persist_1h"] = df["v_obs"].shift(1)
    df["v_persist_6h"] = df["v_obs"].shift(6)
    df["v_persist_24h"] = df["v_obs"].shift(24)
    df["v_persist_27d"] = df["v_obs"].shift(27 * 24)

    # Persistence deltas.
    df["delta_v_1h_6h"] = df["v_persist_1h"] - df["v_persist_6h"]
    df["delta_v_1h_24h"] = df["v_persist_1h"] - df["v_persist_24h"]
    df["delta_v_24h_27d"] = df["v_persist_24h"] - df["v_persist_27d"]

    # Plasma context at issue_time.
    df["abs_bz"] = df["bz_obs"].abs()
    df["southward_bz"] = np.maximum(0.0, -df["bz_obs"])

    # Solar rotation phase at issue_time.
    unix_hours = df.index.view("int64") / 1e9 / 3600.0
    period_hours = SOLAR_ROTATION_DAYS * 24.0

    df["issue_sin_27d"] = np.sin(2.0 * np.pi * unix_hours / period_hours)
    df["issue_cos_27d"] = np.cos(2.0 * np.pi * unix_hours / period_hours)

    df = df.reset_index()

    keep = ["issue_time"] + BASE_FEATURE_COLUMNS

    df = df[keep]

    # For training rows, persistence must exist.
    df = df.dropna(
        subset=[
            "v_obs",
            "v_persist_1h",
            "v_persist_6h",
            "v_persist_24h",
            "v_persist_27d",
        ]
    )

    return df

def _build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fast lookup for target_v.
    target_lookup = (
        df[["issue_time", "v_obs"]]
        .rename(columns={
            "issue_time": "valid_time",
            "v_obs": "target_v",
        })
        .set_index("valid_time")
    )

    rows = []

    for _, row in df.iterrows():

        issue_time = row["issue_time"]

        for lead_hours in range(1, MAX_LEAD_HOURS + 1):

            valid_time = issue_time + pd.Timedelta(hours=lead_hours)

            if valid_time not in target_lookup.index:
                continue

            target_v = target_lookup.loc[valid_time, "target_v"]

            sample = {
                "issue_time": issue_time,
                "valid_time": valid_time,
                "lead_hours": lead_hours,
                "lead_norm": lead_hours / MAX_LEAD_HOURS,
                "target_v": float(target_v),
            }

            # Copy issue-time features.
            for col in BASE_FEATURE_COLUMNS:
                sample[col] = row[col]

            # Valid-time solar rotation phase.
            unix_hours = valid_time.timestamp() / 3600.0
            period_hours = SOLAR_ROTATION_DAYS * 24.0

            sample["valid_sin_27d"] = np.sin(
                2.0 * np.pi * unix_hours / period_hours
            )

            sample["valid_cos_27d"] = np.cos(
                2.0 * np.pi * unix_hours / period_hours
            )

            # Binary event targets.
            sample["target_high_speed_450"] = int(target_v >= 450)
            sample["target_high_speed_500"] = int(target_v >= 500)
            sample["target_high_speed_600"] = int(target_v >= 600)
            sample["target_high_speed_700"] = int(target_v >= 700)
            sample["target_high_speed_800"] = int(target_v >= 800)

            rows.append(sample)

    out = pd.DataFrame(rows)

    out = out.sort_values([
        "issue_time",
        "lead_hours",
    ])

    return out

def main():
    parser = argparse.ArgumentParser(
        description="OMNIWeb historical data fetch"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="omniweb observations file, csv",
        default=Path("./data/historical/omni.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output file",
        default=Path("./data/historical/omni_processed.csv")
    )
    
    args = parser.parse_args()

    df = _build_features(
        source=args.input
    )

    df = _build_dataset(df)

    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()