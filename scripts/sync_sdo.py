#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import drms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install the JSOC client first: uv pip install drms") from exc

"""
To get access to observation data, You should register on 
http://jsoc.stanford.edu/ajax/register_email.html
and confirm your email
"""

""""
More info on page
http://jsoc.stanford.edu/AIA/AIA_jsoc.html
and
http://jsoc.stanford.edu/ajax/lookdata.html?ds=aia.lev1_euv_12s
this series includes data corresponding to wavelengths 94,131,171,193,211,304 and 335 Å
http://jsoc.stanford.edu/ajax/lookdata.html?ds=aia.lev1_euv_24s
also contains 1600 Å
"""
AIA_MAP = {
    "aia94": 94,
    "aia131": 131,
    "aia171": 171,
    "aia193": 193,
    "aia211": 211,
    "aia304": 304,
    "aia335": 335,
    #"aia1600": 1600,
}

"""
More info on pages
http://jsoc.stanford.edu/ajax/lookdata.html?ds=hmi.M_720s
http://jsoc.stanford.edu/ajax/lookdata.html?ds=hmi.B_720s
etc.
"""
HMI_MAP = {
    "hmi_m": ("hmi.M_720s", "magnetogram"),
    "hmi_bx": ("hmi.B_720s", "field"),
    "hmi_by": ("hmi.B_720s", "inclination"),
    "hmi_bz": ("hmi.B_720s", "azimuth"),
    "hmi_v": ("hmi.V_720s", "Dopplergram"),
}


def _build_aia_query(ts: pd.Timestamp, wavelengths: list, segment: str = "image") -> str:
    stamp = ts.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    wavelength = ",".join(map(str, wavelengths))
    return f"aia.lev1_euv_12s[{stamp}][{wavelength}]{{{segment}}}"


def _build_hmi_query(ts: pd.Timestamp, series: str, segment: str) -> str:
    stamp = ts.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    return f"{series}[{stamp}]{{{segment}}}"

def _file_already_present(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0

def import_records(
    email: str,
    timestamps: Iterable[pd.Timestamp],
    out_dir: Path,
    method: str = "url_quick",
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    client = drms.Client(email=email)
    out_dir = Path(out_dir)

    manifest_lookup: set[tuple[str, str]] = set()
    existing_manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else None
    if existing_manifest is not None and not existing_manifest.empty:
        manifest = existing_manifest.copy()
        manifest_lookup = {
            (row["timestamp"], row["channel"])
            for _, row in manifest.iterrows()
            if Path(row["path"]).exists()
        }
    else:
        manifest = pd.DataFrame(columns=["timestamp", "channel", "path", "status"])
    
    def normalize_ts(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")

    def already_have(ts: str, channel: str) -> bool:
        return (ts, channel) in manifest_lookup
    
    for raw_ts in timestamps:
        ts = normalize_ts(raw_ts)
        tai_ts = ts.tz_convert("UTC")

        missed_wavelengths = []
        downloaded = None

        for channel, wavelength in AIA_MAP.items():
            if already_have(ts.isoformat(), channel):
                continue

            channel_dir = out_dir / channel
            channel_dir.mkdir(parents=True, exist_ok=True)

            missed_wavelengths.append(wavelength)
        
        if len(missed_wavelengths) > 0:
            query = _build_aia_query(tai_ts, missed_wavelengths)
            req = client.export(query, method=method, protocol="fits")
            req.wait()

            downloaded = req.download(str(channel_dir))
        
        if downloaded is not None:
            for f in downloaded.download:
                f = Path(f)
                if f.is_file() and f.stat().st_size > 0:
                    ns = f.name.split(".")
                    if len(ns) < 4:
                        # Happened if no data presented. Old data
                        # or, more important, NEW DATA
                        # Should try to get them some other way
                        continue
                    wavelength = ns[-3]
                    manifest.loc[len(manifest)] = [
                        ts.isoformat(),
                        f"aia{wavelength}",
                        str(f),
                        "downloaded"
                    ]
                    manifest.to_parquet(manifest_path)

        for channel, (series, segment) in HMI_MAP.items():
            continue
            if already_have(ts, channel):
                continue

            channel_dir = out_dir / channel
            channel_dir.mkdir(parents=True, exist_ok=True)

            query = _build_hmi_query(tai_ts, series, segment)
            req = client.export(query, method=method, protocol="fits")
            req.wait()

            downloaded = req.download(str(channel_dir))
            for f in downloaded.download:
                f = Path(f)
                if f.is_file() and f.stat().st_size > 0:
                    manifest.loc[len(manifest)] = [
                        ts.isoformat(),
                        channel,
                        str(f),
                        "downloaded"
                    ]
                    manifest.to_parquet(manifest_path)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SDO AIA/HMI FITS files from JSOC for Argus Surya channels.")
    parser.add_argument("--email", required=True, help="JSOC-registered email address required for export jobs")
    parser.add_argument("--timestamps-csv", type=Path, required=True, help="CSV with a timestamp column")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/sdo"))
    parser.add_argument("--manifest-out", type=Path, default=Path("data/raw/sdo/manifest.parquet"))
    parser.add_argument("--method", default="url_quick", choices=["url", "url_quick", "url-tar", "ftp", "ftp-tar"])
    args = parser.parse_args()

    df = pd.read_csv(args.timestamps_csv)
    if "timestamp" not in df.columns:
        raise SystemExit("timestamps-csv must contain a 'timestamp' column")

    timestamps = pd.to_datetime(df["timestamp"], utc=True).drop_duplicates().sort_values()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_out)

    manifest = import_records(args.email, timestamps, args.out_dir, args.method, manifest_path)
    
    print(f"saved manifest with {len(manifest)} files to {args.manifest_out}")

if __name__ == "__main__":
    main()
