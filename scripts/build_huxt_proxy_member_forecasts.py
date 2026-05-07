#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MEMBERS = 16
DEFAULT_RANDOM_SEED = 42


REQUIRED_COLUMNS = [
    "issue_time",
    "valid_time",
    "lead_hours",

    "v_persist_1h",
    "v_persist_24h",
    "v_persist_27d",

    "delta_v_1h_24h",
    "delta_v_24h_27d",
]


def load_training_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["lead_hours"] = df["lead_hours"].astype(int)

    for col in REQUIRED_COLUMNS:
        if col not in ["issue_time", "valid_time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.sort_values(["issue_time", "lead_hours"])

    return df


def lead_weight(lead_hours: int) -> float:
    return min(1.0, max(0.0, lead_hours / 96.0))


def build_proxy_members(
    df: pd.DataFrame,
    members: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []

    for _, row in df.iterrows():
        issue_time = row["issue_time"]
        valid_time = row["valid_time"]
        lead_hours = int(row["lead_hours"])

        v_recent = float(row["v_persist_1h"])
        v_daily = float(row["v_persist_24h"])
        v_recurrent = float(row["v_persist_27d"])

        recent_trend = float(row["delta_v_1h_24h"])
        recurrence_delta = float(row["delta_v_24h_27d"])

        w = lead_weight(lead_hours)

        for member_id in range(members):
            recurrent_offset = rng.normal(0.0, 35.0)
            trend_scale = rng.normal(1.0, 0.25)
            relaxation_scale = rng.normal(1.0, 0.15)

            base = (
                (1.0 - w) * v_recent
                + w * (v_recurrent + recurrent_offset)
            )

            trend_decay = np.exp(-lead_hours / 24.0)
            trend_term = trend_scale * recent_trend * trend_decay * 0.25

            relaxation_term = (
                relaxation_scale
                * recurrence_delta
                * (1.0 - np.exp(-lead_hours / 36.0))
                * 0.10
            )

            # Small member noise increasing with horizon.
            horizon_noise = rng.normal(
                0.0,
                5.0 + 0.25 * lead_hours,
            )

            v_huxt = base + trend_term + relaxation_term + horizon_noise
            v_huxt = float(np.clip(v_huxt, 250.0, 950.0))

            rows.append({
                "issue_time": issue_time,
                "valid_time": valid_time,
                "lead_hours": lead_hours,
                "member_id": member_id,
                "v_huxt": v_huxt,
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build proxy HUXt-like member forecasts from existing training rows."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="training_dataset.csv with one row per issue_time/lead_hours.",
    )

    parser.add_argument(
        "--output",
        default=Path("data/processed/huxt_member_forecasts.csv"),
        type=Path,
    )

    parser.add_argument(
        "--members",
        default=DEFAULT_MEMBERS,
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=DEFAULT_RANDOM_SEED,
        type=int,
    )

    args = parser.parse_args()

    dataset = load_training_dataset(args.dataset)

    forecasts = build_proxy_members(
        dataset,
        members=args.members,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(args.output, index=False)

    expected = len(dataset) * args.members

    print(f"input dataset rows: {len(dataset)}")
    print(f"members: {args.members}")
    print(f"expected rows: {expected}")
    print(f"saved rows: {len(forecasts)}")
    print(f"output: {args.output}")
    print()
    print(forecasts.head())


if __name__ == "__main__":
    main()