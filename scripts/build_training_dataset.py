
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from forecast.data_pipelines.feature_building import build_features, BASE_FEATURE_COLUMNS

SOLAR_ROTATION_DAYS = 27.2753
MAX_LEAD_HOURS = 96

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

    for _, row in tqdm(df.iterrows(), total=len(df)):

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
        default=Path("./data/raw/omni.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output file",
        default=Path("./data/processed/omni_features.csv")
    )
    
    args = parser.parse_args()

    print("Reading raw observations file...")
    df = pd.read_csv(args.input, parse_dates=["issue_time"])

    print("Building extra features...")
    df = build_features(df)

    print("Building training set with 96-hours samples...")
    df = _build_dataset(df)

    df.to_csv(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()