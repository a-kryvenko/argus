"""Loader for live GOES-R solar irradiance observations.

The normalized records contain the EUVS measurements needed to reconstruct
S10 and M10, plus the XRS background and Lyman-alpha measurements needed to
reconstruct Y10. The reconstruction and calibration themselves belong in
forecast-core rather than in this source-data loader.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

GOES_EUVS_URL = "https://services.swpc.noaa.gov/json/goes/primary/euvs-1-day.json"
GOES_XRAY_BACKGROUND_URL = (
    "https://services.swpc.noaa.gov/json/goes/primary/"
    "xray-background-7-day.json"
)

REQUEST_TIMEOUT_SECONDS = 120

EUVS_LINE_COLUMNS = {
    "256": "goes_euv_256",
    "284": "goes_euv_284",
    "304": "goes_euv_304",
    "1175": "goes_euv_1175",
    "1216": "goes_lya_1216",
    "1335": "goes_euv_1335",
    "1405": "goes_euv_1405",
    "mgii_index": "goes_mgii_index",
}

QUALITY_FLAGS = ("eclipse", "lunar_transit", "geocorona")


def _parse_euvs_json(payload: Any) -> pd.DataFrame:
    """Normalize the SWPC one-minute EUVS response into one row per timestamp."""
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("GOES EUVS response is empty or malformed")

    rows = []
    for record in payload:
        if not isinstance(record, dict):
            continue

        line = str(record.get("line", ""))
        if line not in EUVS_LINE_COLUMNS:
            continue

        try:
            value = float(record["value"])
            satellite = int(record["satellite"])
            au_factor = float(record["au_factor"])
        except (KeyError, TypeError, ValueError):
            continue

        timestamp = pd.to_datetime(record.get("time_tag"), utc=True, errors="coerce")
        if pd.isna(timestamp) or not np.isfinite(value) or not np.isfinite(au_factor):
            continue

        flags = record.get("flags")
        flags = flags if isinstance(flags, dict) else {}
        rows.append({
            "timestamp": timestamp,
            "goes_euvs_satellite": satellite,
            "goes_au_factor": au_factor,
            "line": line,
            "value": value,
            **{
                f"goes_{flag}": bool(flags.get(flag, False))
                for flag in QUALITY_FLAGS
            },
        })

    if not rows:
        raise RuntimeError("No valid GOES EUVS observations parsed")

    records = pd.DataFrame(rows)
    value_frame = records.pivot_table(
        index=["timestamp", "goes_euvs_satellite"],
        columns="line",
        values="value",
        aggfunc="last",
    ).rename(columns=EUVS_LINE_COLUMNS)

    required_columns = list(EUVS_LINE_COLUMNS.values())
    value_frame = value_frame.reindex(columns=required_columns).dropna(
        subset=required_columns
    )
    if value_frame.empty:
        raise RuntimeError("GOES EUVS response contains no complete observation")

    metadata = records.groupby(
        ["timestamp", "goes_euvs_satellite"],
        sort=True,
    ).agg({
        "goes_au_factor": "last",
        **{f"goes_{flag}": "max" for flag in QUALITY_FLAGS},
    })

    frame = value_frame.join(metadata).reset_index()
    frame.columns.name = None
    frame["goes_euvs_quality_valid"] = ~frame[
        [f"goes_{flag}" for flag in QUALITY_FLAGS]
    ].any(axis=1)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _parse_xray_background_json(payload: Any) -> pd.DataFrame:
    """Normalize the SWPC daily XRS background response."""
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("GOES X-ray background response is empty or malformed")

    rows = []
    for record in payload:
        if not isinstance(record, dict):
            continue

        try:
            background = float(record["background"])
            satellite = int(record["satellite"])
        except (KeyError, TypeError, ValueError):
            continue

        timestamp = pd.to_datetime(record.get("time_tag"), utc=True, errors="coerce")
        if pd.isna(timestamp) or not np.isfinite(background):
            continue

        rows.append({
            "goes_xray_background_timestamp": timestamp,
            "goes_xray_satellite": satellite,
            "goes_xray_background": background,
        })

    if not rows:
        raise RuntimeError("No valid GOES X-ray background observations parsed")

    return (
        pd.DataFrame(rows)
        .drop_duplicates("goes_xray_background_timestamp", keep="last")
        .sort_values("goes_xray_background_timestamp")
        .reset_index(drop=True)
    )


def _fetch_json(url: str) -> Any:
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _utc_timestamp(value: datetime.datetime | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def fetch_goes(
    start: datetime.datetime | str | None = None,
    end: datetime.datetime | str | None = None,
    *,
    euvs_url: str = GOES_EUVS_URL,
    xray_background_url: str = GOES_XRAY_BACKGROUND_URL,
) -> pd.DataFrame:
    """Fetch live GOES-R EUVS/XRS observations and optionally filter by time.

    X-ray background is a daily product. Each minute-level EUVS row receives
    the most recent background value whose timestamp is not later than the
    EUVS timestamp. ``start`` and ``end`` are inclusive and may be timezone-
    aware or naive; naive values are interpreted as UTC.
    """
    euvs = _parse_euvs_json(_fetch_json(euvs_url))
    xray_background = _parse_xray_background_json(
        _fetch_json(xray_background_url)
    )

    records = pd.merge_asof(
        euvs.sort_values("timestamp"),
        xray_background,
        left_on="timestamp",
        right_on="goes_xray_background_timestamp",
        direction="backward",
    )

    if start is not None:
        records = records.loc[records["timestamp"] >= _utc_timestamp(start)]
    if end is not None:
        records = records.loc[records["timestamp"] <= _utc_timestamp(end)]

    return records.reset_index(drop=True)
