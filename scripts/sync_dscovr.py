#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

"""
Service provide real time solar weather observations (plasma, mag, etc.)
"""
BASE = "https://services.swpc.noaa.gov/products/solar-wind"

PLASMA_FILES = {
    "2h": "plasma-2-hour.json",
    "6h": "plasma-6-hour.json",
    "1d": "plasma-1-day.json",
    "3d": "plasma-3-day.json",
    "7d": "plasma-7-day.json",
}
MAG_FILES = {
    "2h": "mag-2-hour.json",
    "6h": "mag-6-hour.json",
    "1d": "mag-1-day.json",
    "3d": "mag-3-day.json",
    "7d": "mag-7-day.json",
}


def _fetch_json(url: str) -> list[list[object]]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected JSON payload from {url}")
    return data


def _rows_to_frame(rows: list[list[object]]) -> pd.DataFrame:
    header = rows[0]
    values = rows[1:]
    df = pd.DataFrame(values, columns=header)
    for candidate in ["time_tag", "time_tag_utc", "time_tag_r"]:
        if candidate in df.columns:
            df.insert(0, column="timestamp", value=pd.to_datetime(df[candidate], utc=True, errors="coerce"))
            df = df.drop(candidate, axis=1)
            break
    if "timestamp" not in df.columns:
        raise ValueError(f"No time column in JSON header: {header}")
    df.set_index("timestamp")
    return df


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_frame(range_key: str) -> pd.DataFrame:
    plasma = _rows_to_frame(_fetch_json(f"{BASE}/{PLASMA_FILES[range_key]}"))
    mag = _rows_to_frame(_fetch_json(f"{BASE}/{MAG_FILES[range_key]}"))

    plasma = _coerce_numeric(plasma, ["density", "speed", "temperature"])
    mag = _coerce_numeric(mag, ["bx_gsm", "by_gsm", "bz_gsm", "lon_gsm", "lat_gsm", "bt"])

    keep_plasma = [c for c in ["timestamp", "density", "speed", "temperature"] if c in plasma.columns]
    keep_mag = [c for c in ["timestamp", "bx_gsm", "by_gsm", "bz_gsm"] if c in mag.columns]

    merged = plasma[keep_plasma].merge(mag[keep_mag], on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    rename_map = {
        "bx_gsm": "BX_GSM",
        "by_gsm": "BY_GSM",
        "bz_gsm": "BZ_GSM",
        "speed": "V",
        "density": "N",
        "temperature": "T"
    }
    merged = merged.rename(columns=rename_map)

    cols = [c for c in ["timestamp", "BX_GSM", "BY_GSM", "BZ_GSM", "V", "N", "T"] if c in merged.columns]
    return merged[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NOAA SWPC real-time DSCOVR solar wind JSON and save as CSV/Parquet.")
    parser.add_argument("--range", choices=PLASMA_FILES.keys(), default="7d")
    parser.add_argument("--out", type=Path, default=Path("data/raw/dscovr/dscovr_7d.parquet"))
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    df = build_frame(args.range)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        df.to_csv(args.out.with_suffix(".csv"), index=False)
        print(f"saved {len(df)} rows to {args.out.with_suffix('.csv')}")
    else:
        df.to_parquet(args.out)
        print(f"saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
