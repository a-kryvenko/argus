#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


HUXT_FEATURE_COLUMNS = [
    "ens_mean",
    "ens_std",
    "ens_min",
    "ens_max",
    "ens_p10",
    "ens_p50",
    "ens_p90",
    "ens_members",
    "ens_spread",
    "ens_skew_proxy",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge HUXt ensemble features into speed forecast training dataset."
    )

    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--huxt-features", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=Path("data/processed/training_dataset_with_huxt.csv"),
        type=Path,
    )

    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    huxt = pd.read_csv(args.huxt_features)

    for df in [dataset, huxt]:
        df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        df["lead_hours"] = df["lead_hours"].astype(int)

    merged = dataset.merge(
        huxt,
        on=["issue_time", "valid_time", "lead_hours"],
        how="inner",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"dataset rows: {len(dataset)}")
    print(f"huxt rows: {len(huxt)}")
    print(f"merged rows: {len(merged)}")
    print(f"saved to {args.output}")

    coverage = len(merged) / len(dataset) * 100.0
    print(f"HUXt coverage: {coverage:.2f}%")

    print()
    print("HUXt features:")
    print(HUXT_FEATURE_COLUMNS)


if __name__ == "__main__":
    main()