#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

"""
https://omniweb.gsfc.nasa.gov/html/overview.html

The OMNI 2 data set was created at NSSDC in 2003 as a successor to the OMNI data set first created in the mid-1970's.
The OMNI 2 data set contains hourly resolution solar wind magnetic field and plasma data from many
spacecraft in geocentric orbit and in orbit about the L1 Lagrange point ~225 Re in front of the Earth.
The data set also contains hourly fluxes of energetic protons, geomagnetic activity indices (AE, Dst, etc.)
and sunspot numbers. The data set is periodically updated.
Details about the contents and preparation of OMNI 2 are found in the two other options
of the "About OMNI 2 data and OMNIWeb interface" section of the top OMNIWeb page.

Omni do not provide latest observations, so it's not unexpected to not have observations for this month, for example
"""

OMNI_URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

"""
https://omniweb.gsfc.nasa.gov/form/dx1.html

Example of response. Keep in mind that numbers is relative and build based on requested columns:
<HTML>
<HEAD><TITLE>OMNIWeb Results</TITLE></HEAD>
<BODY>
<center><font size=5 color=red>OMNIWeb Plus Browser Results </font></center><br>
<B>Listing for omni2 data from 20260301 to 20260301</B><hr><pre>Selected parameters:
 1 BY, nT (GSM)
 2 BZ, nT (GSM)
 3 SW Plasma Speed, km/s
 4 SW Proton Density, N/cm^3

YEAR DOY HR    1     2     3     4 
2026  60  0  -2.0  -2.3  388.   3.4
"""


"""
Variable ids based on OMNIWeb's command-line interface for HOURLY OMNI.
https://omniweb.gsfc.nasa.gov/form/dx1.html
"""
DEFAULT_VARS = {
    "BX_GSM": {"id": 12, "title": "Bx, GSE/GSM, nT"},
    "BY_GSM": {"id": 15, "title": "BY, nT (GSM)"},
    "BZ_GSM": {"id": 16, "title": "BZ, nT (GSM)"},
    "V": {"id": 24, "title": "SW Plasma Speed, km/s"},
    "N": {"id": 23, "title": "SW Proton Density, N/cm^3"},
    "T": {"id": 22, "title": "SW Plasma Temperature, K"}
}

def _parse_omni_text(text: str) -> pd.DataFrame:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if (
            parts[0].isdigit()
            and len(parts[0]) == 4 # Year column in text
            and parts[1].isdigit() # DOY (day of year) column in text
            and 1 <= len(parts[1]) <= 3
            and parts[2].isdigit() # HR (hour) column in text
            and 0 <= int(parts[2]) <= 23
        ):
            rows.append(parts)

    if not rows:
        raise RuntimeError("No OMNI records parsed; inspect response format or variable ids.")

    cols = ["year", "doy", "hr", *DEFAULT_VARS.keys()]
    trimmed = [r[: len(cols)] for r in rows if len(r) >= len(cols)]
    df = pd.DataFrame(trimmed, columns=cols)

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ts = (
        pd.to_datetime(
            df["year"].astype(int).astype(str)
            + df["doy"].astype(int).astype(str).str.zfill(3),
            format="%Y%j",
            utc=True,
        )
        + pd.to_timedelta(df["hr"].astype(int), unit="h")
    )
    df["timestamp"] = ts
    df = df.drop(columns=["year", "doy", "hr"])

    # OMNI missing observations are large numbers like 9999.9 or 9999.99
    for c in DEFAULT_VARS.keys():
        mask = (df[c].abs() >= 9_999) & (df[c].abs() <= 10_000)
        df.loc[mask, c] = pd.NA

    return df[["timestamp", "BX_GSM", "BY_GSM", "BZ_GSM", "V", "N", "T"]]


def fetch_omni(start: str, end: str) -> pd.DataFrame:
    payload = {
        "activity": "retrieve",
        "res": "hour",
        "scale": "Linear",
        "spacecraft": "omni2",
        "start_date": start,
        "end_date": end,
        "table": 0,
    }
    for param in DEFAULT_VARS.values():
        payload.setdefault("vars", [])
        payload["vars"].append(str(param.get("id")))

    response = requests.post(OMNI_URL, data=payload, timeout=120)
    response.raise_for_status()

    return _parse_omni_text(response.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download hourly OMNI data from NASA OMNIWeb.")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--out", type=Path, default=Path("data/raw/omni/omni_hourly.parquet"))
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    df = fetch_omni(args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        path = args.out.with_suffix(".csv")
        df.to_csv(path, index=False)
    else:
        path = args.out
        df.to_parquet(path)

    print(f"saved {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
