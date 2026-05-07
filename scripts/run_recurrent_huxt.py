#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REQUIRED_COLUMNS = [
    "issue_time",
    "valid_time",
    "lead_hours",
    "v_persist_1h",
    "v_persist_6h",
    "v_persist_24h",
    "v_persist_27d",
    "delta_v_1h_24h",
    "delta_v_24h_27d",
]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    df["lead_hours"] = df["lead_hours"].astype(int)

    for col in REQUIRED_COLUMNS:
        if col not in {"issue_time", "valid_time"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df.sort_values(["issue_time", "lead_hours"])

    return df


def make_recurrent_boundary(row: pd.Series, member_rng: np.random.Generator) -> dict:
    """
    Minimal recurrent-boundary approximation.

    This is not WSA.
    It creates a smooth inner-boundary speed prior from:
      - recent speed
      - 24h persistence
      - 27d recurrence
      - trend terms

    Later this function is the only part we replace with real WSA boundary input.
    """

    v_recent = float(row["v_persist_1h"])
    v_6h = float(row["v_persist_6h"])
    v_24h = float(row["v_persist_24h"])
    v_27d = float(row["v_persist_27d"])

    trend_24h = float(row["delta_v_1h_24h"])
    recurrence_delta = float(row["delta_v_24h_27d"])

    member = {
        "v_recent": v_recent + member_rng.normal(0.0, 10.0),
        "v_6h": v_6h + member_rng.normal(0.0, 12.0),
        "v_24h": v_24h + member_rng.normal(0.0, 18.0),
        "v_27d": v_27d + member_rng.normal(0.0, 35.0),
        "trend_24h": trend_24h * member_rng.normal(1.0, 0.25),
        "recurrence_delta": recurrence_delta * member_rng.normal(1.0, 0.25),
    }

    return member


def run_simple_recurrent_propagation(
    boundary: dict,
    lead_hours: int,
) -> float:
    """
    Placeholder for real HUXt call.

    This function deliberately has the same interface we will later keep:

        boundary + lead_hours -> v_huxt

    For now it is a deterministic reduced propagation approximation.
    Next step is replacing its internals with actual huxt.HUXt model calls.
    """

    lead = float(lead_hours)

    # Transition from near-term observed state to recurrent state.
    w_recurrent = np.clip(lead / 96.0, 0.0, 1.0)

    # Near-term memory decays with lead.
    w_recent = np.exp(-lead / 18.0)

    # 24h state contributes mostly in the 12–48h range.
    w_24h = np.exp(-((lead - 30.0) ** 2) / (2.0 * 24.0 ** 2))

    # Base speed.
    v = (
        0.45 * w_recent * boundary["v_recent"]
        + 0.25 * w_24h * boundary["v_24h"]
        + (0.30 + 0.70 * w_recurrent) * boundary["v_27d"]
    )

    norm = (
        0.45 * w_recent
        + 0.25 * w_24h
        + (0.30 + 0.70 * w_recurrent)
    )

    v = v / max(norm, 1e-6)

    # Smooth trend propagation.
    v += 0.20 * boundary["trend_24h"] * np.exp(-lead / 30.0)

    # Stream-interaction-like relaxation.
    v += 0.10 * boundary["recurrence_delta"] * (1.0 - np.exp(-lead / 36.0))

    return float(np.clip(v, 250.0, 950.0))


def build_member_forecasts(
    dataset: pd.DataFrame,
    members: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = dataset.copy().reset_index(drop=True)
    n = len(df)

    # Repeat each forecast row for each ensemble member
    repeated = df.loc[df.index.repeat(members), [
        "issue_time",
        "valid_time",
        "lead_hours",
        "v_persist_1h",
        "v_persist_6h",
        "v_persist_24h",
        "v_persist_27d",
        "delta_v_1h_24h",
        "delta_v_24h_27d",
    ]].reset_index(drop=True)

    repeated["member_id"] = np.tile(np.arange(members), n)

    lead = repeated["lead_hours"].to_numpy(dtype=np.float32)

    v_recent = repeated["v_persist_1h"].to_numpy(dtype=np.float32)
    v_24h = repeated["v_persist_24h"].to_numpy(dtype=np.float32)
    v_27d = repeated["v_persist_27d"].to_numpy(dtype=np.float32)
    trend_24h = repeated["delta_v_1h_24h"].to_numpy(dtype=np.float32)
    recurrence_delta = repeated["delta_v_24h_27d"].to_numpy(dtype=np.float32)

    size = len(repeated)

    v_recent = v_recent + rng.normal(0.0, 10.0, size)
    v_24h = v_24h + rng.normal(0.0, 18.0, size)
    v_27d = v_27d + rng.normal(0.0, 35.0, size)

    trend_24h = trend_24h * rng.normal(1.0, 0.25, size)
    recurrence_delta = recurrence_delta * rng.normal(1.0, 0.25, size)

    w_recurrent = np.clip(lead / 96.0, 0.0, 1.0)
    w_recent = np.exp(-lead / 18.0)
    w_24h = np.exp(-((lead - 30.0) ** 2) / (2.0 * 24.0 ** 2))

    v = (
        0.45 * w_recent * v_recent
        + 0.25 * w_24h * v_24h
        + (0.30 + 0.70 * w_recurrent) * v_27d
    )

    norm = (
        0.45 * w_recent
        + 0.25 * w_24h
        + (0.30 + 0.70 * w_recurrent)
    )

    v = v / np.maximum(norm, 1e-6)

    v += 0.20 * trend_24h * np.exp(-lead / 30.0)
    v += 0.10 * recurrence_delta * (1.0 - np.exp(-lead / 36.0))

    repeated["v_huxt"] = np.clip(v, 250.0, 950.0)

    return repeated[
        [
            "issue_time",
            "valid_time",
            "lead_hours",
            "member_id",
            "v_huxt",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run recurrent-boundary HUXt-style ensemble forecast."
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

    parser.add_argument("--members", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    forecasts = build_member_forecasts(
        dataset=dataset,
        members=args.members,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(args.output, index=False)

    print(f"input rows: {len(dataset)}")
    print(f"members: {args.members}")
    print(f"saved rows: {len(forecasts)}")
    print(f"output: {args.output}")
    print(forecasts.head())


if __name__ == "__main__":
    main()