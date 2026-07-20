from __future__ import annotations

import datetime

import pandas as pd
import requests


"""Loader for the solar indices used by the JB2008 atmosphere model.

Source and format documentation:
https://spacewx.com/jb2008/
https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT

The public SET file is historical and is normally released with a delay.  It
contains daily F10, S10, M10 and Y10 values plus their centered 81-day means.
"""

SPACEWX_SOLFSMY_URL = "https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT"

SOLFSMY_COLUMNS = [
    "year",
    "day_of_year",
    "julian_day",
    "f10",
    "f10_81c",
    "s10",
    "s10_81c",
    "m10",
    "m10_81c",
    "y10",
    "y10_81c",
    "source",
]

INDEX_COLUMNS = SOLFSMY_COLUMNS[3:11]


def _parse_spacewx_text(text: str) -> pd.DataFrame:
    """Parse a SET SOLFSMY text file into normalized daily records."""
    rows: list[list[str]] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if (
            len(parts) >= 11
            and parts[0].isdigit()
            and len(parts[0]) == 4
            and parts[1].isdigit()
            and 1 <= int(parts[1]) <= 366
        ):
            # Some releases omit the optional source flag.
            rows.append(parts[:12] + [""] * max(0, 12 - len(parts)))

    if not rows:
        raise RuntimeError("No SOLFSMY records parsed; inspect the SET response format.")

    frame = pd.DataFrame(rows, columns=SOLFSMY_COLUMNS)
    numeric_columns = SOLFSMY_COLUMNS[:-1]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    if frame[numeric_columns].isna().any().any():
        invalid_rows = frame.index[frame[numeric_columns].isna().any(axis=1)].tolist()
        raise RuntimeError(f"Invalid numeric values in SOLFSMY rows: {invalid_rows[:5]}")

    frame["timestamp"] = pd.to_datetime(
        frame["year"].astype(int).astype(str)
        + frame["day_of_year"].astype(int).astype(str).str.zfill(3),
        format="%Y%j",
        utc=True,
    ) + pd.Timedelta(hours=12)

    if frame["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate dates found in SOLFSMY response.")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame[["timestamp", *SOLFSMY_COLUMNS]]


def fetch_spacewx(
    start: datetime.datetime | str | None = None,
    end: datetime.datetime | str | None = None,
    *,
    url: str = SPACEWX_SOLFSMY_URL,
) -> pd.DataFrame:
    """Download SOLFSMY history and optionally restrict it to an inclusive range."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    records = _parse_spacewx_text(response.text)

    if start is not None:
        start_ts = pd.Timestamp(start)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        records = records.loc[records["timestamp"] >= start_ts]

    if end is not None:
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        records = records.loc[records["timestamp"] <= end_ts]

    return records.reset_index(drop=True)
