import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
import cdflib
import tempfile

import pandas as pd
import numpy as np

from common.config import get_config
from common.schema import ObservationPoint, Observation

DSCOVR_2H_PLASMA_BASE_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-"
DSCOVR_2H_MAG_BASE_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-"
GONG_MAG_URL = "https://services.swpc.noaa.gov/products/gong/"
KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SOLAR_CYCLE_INFO_URL = "https://services.swpc.noaa.gov/products/solar-cycle-25-f10-7-predicted-range.json"

COLUMNS = [
    "issue_time",
    "bx",
    "by",
    "bz",
    "v",
    "n",
    "t",
    "kp"
]

def _fetch_latest_observations(endpoint: str) -> pd.DataFrame:
    df = _fetch_live_mag(endpoint)
    df = df.merge(_fetch_live_plasma(endpoint), how="left", on="issue_time")
    df = df.merge(_fetch_live_kp(), how="left", on="issue_time")

    df = df.set_index("issue_time", drop=False)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.resample("1h").first()

    df = df[COLUMNS]

    return df

def _fetch_live_mag(endpoint: str) -> pd.DataFrame:
    r = requests.get(DSCOVR_2H_MAG_BASE_URL + endpoint)
    r.raise_for_status()
    data = r.json()

    mag_df = pd.DataFrame(data[1:], columns=data[0])
    mag_df["issue_time"] = pd.to_datetime(mag_df["time_tag"])
    mag_df["issue_time"] = mag_df["issue_time"].dt.tz_localize("UTC")

    mag_df["bx"] = pd.to_numeric(mag_df["bx_gsm"])
    mag_df["by"] = pd.to_numeric(mag_df["by_gsm"])
    mag_df["bz"] = pd.to_numeric(mag_df["bz_gsm"])

    return mag_df

def _fetch_live_plasma(endpoint: str) -> pd.DataFrame:
    r = requests.get(DSCOVR_2H_PLASMA_BASE_URL + endpoint)
    r.raise_for_status()
    data = r.json()

    plasma_df = pd.DataFrame(data[1:], columns=data[0])
    plasma_df["issue_time"] = pd.to_datetime(plasma_df["time_tag"])
    plasma_df["issue_time"] = plasma_df["issue_time"].dt.tz_localize("UTC")
    plasma_df["n"] = pd.to_numeric(plasma_df["density"])
    plasma_df["v"] = pd.to_numeric(plasma_df["speed"])
    plasma_df["t"] = pd.to_numeric(plasma_df["temperature"])

    return plasma_df

def _fetch_live_kp() -> pd.DataFrame:
    r = requests.get(KP_INDEX_URL)
    r.raise_for_status()
    data = r.json()
    kp_df = pd.DataFrame(data, columns=["time_tag", "Kp"])
    kp_df["issue_time"] = pd.to_datetime(kp_df["time_tag"])
    kp_df["issue_time"] = kp_df["issue_time"].dt.tz_localize("UTC")
    kp_df["kp"] = pd.to_numeric(kp_df["Kp"])
    kp_df = kp_df[["issue_time", "kp"]]
    kp_df = kp_df.set_index("issue_time")

    kp_df = (
        kp_df
        .resample("1h")
        .interpolate(method="time")
        .reset_index()
    )

    return kp_df


def _get_raw_dataset(raw_dataset_path: Path) -> pd.DataFrame:
    now = datetime.now(timezone.utc)

    df = None

    if raw_dataset_path.is_file():
        df = pd.read_csv(raw_dataset_path, parse_dates=["issue_time"])
        df = df.set_index("issue_time", drop=False)
        time_delta = now - df.iloc[-1]["issue_time"]

        # We are able to get only up to 6 day live observations directly from NOAA json products.
        # For older data we must re-download Archive dataset
        if time_delta.total_seconds() > 3600 * 24 * 6:
            df = None
    
    if df is None:
        df = _download_raw_dataset()
    
    time_delta = now - df.iloc[-1]["issue_time"]
    seconds = time_delta.total_seconds()

    if seconds < 60 * 5:
        endpoint = "5-minute.json"
        endpoint = ""
    elif seconds < 3600 * 2:
        endpoint = "2-hour.json"
        endpoint = ""
    elif seconds < 3600 * 6:
        endpoint = "6-hour.json"
    elif seconds < 3600 * 24:
        endpoint = "1-day.json"
    else:
        endpoint = "7-day.json"
    
    if endpoint != "":
        latest_df = _fetch_latest_observations(endpoint=endpoint)
        df = pd.concat([df, latest_df])        

    df.set_index("issue_time", drop=False)
    df = df.sort_index()

    df = df.resample("1h").first()

    df = df.replace('', np.nan)
    df = df.dropna()

    df.to_csv(raw_dataset_path, index=False)

    return df

def _download_raw_dataset() -> pd.DataFrame:
    end_date = datetime.now(timezone.utc) - timedelta(days=6)
    start_date = end_date - timedelta(days=30)

    def load_cdf_from_url(url: str):
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print("missing:", url)
            return None
        
        tmp = tempfile.NamedTemporaryFile(suffix=".cdf", delete=True)
        tmp.write(r.content)
        tmp.flush()

        return cdflib.CDF(tmp.name)
    
    def mag_dataframe(cdf):
        times = cdflib.cdfepoch.to_datetime(
            cdf.varget("Epoch")
        )
        bgse = cdf.varget("BGSEc")
        df = pd.DataFrame({
            "issue_time": times,
            "bx": bgse[:, 0],
            "by": bgse[:, 1],
            "bz": bgse[:, 2],
        })
        df["issue_time"] = df["issue_time"].dt.tz_localize("UTC")
        return df


    def swepam_dataframe(cdf):
        times = cdflib.cdfepoch.to_datetime(
            cdf.varget("Epoch")
        )
        df = pd.DataFrame({
            "issue_time": times,
            "v": cdf.varget("Vp"),
            "n": cdf.varget("Np"),
            "t": cdf.varget("Tpr")
        })
        df["issue_time"] = df["issue_time"].dt.tz_localize("UTC")
        return df
    
    mag_frames = []
    swe_frames = []

    d = start_date

    while d <= end_date:
        year = d.strftime("%Y")
        ymd = d.strftime("%Y%m%d")

        swe_url = f"https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_k0/{year}/ac_k0_swe_{ymd}_v01.cdf"
        swe_cdf = load_cdf_from_url(swe_url)
    
        mag_url = f"https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_k0/{year}/ac_k0_mfi_{ymd}_v01.cdf"
        mag_cdf = load_cdf_from_url(mag_url)

        if mag_cdf and swe_cdf:
            swe_frames.append(
                swepam_dataframe(swe_cdf)
            )
            mag_frames.append(
                mag_dataframe(mag_cdf)
            )

        d += timedelta(days=1)
    
    mag_df = pd.concat(
        mag_frames,
        ignore_index=True
    )

    swe_df = pd.concat(
        swe_frames,
        ignore_index=True
    )

    df = swe_df.merge(mag_df, how="left", on="issue_time")
    df = df.merge(_fetch_live_kp(), how="left", on="issue_time")
    df = df.dropna(
        subset=["issue_time"] + COLUMNS,
        how="all"
    )
    df = df.dropna(subset=["issue_time"])
    df = df.set_index("issue_time", drop=False)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.resample("1h").first()

    df = df[COLUMNS]

    return df


def get_live_observations() -> Observation:
    config = get_config()

    live_dataset = _get_raw_dataset(raw_dataset_path=config.data_root / "raw/live_sensors.csv")

    points = []
    
    for _, record in live_dataset.iterrows():
        points.append(ObservationPoint(
            issue_time=record["issue_time"].to_pydatetime(),
            bx=record["bx"],
            by=record["by"],
            bz=record["bz"],
            v=record["v"],
            n=record["n"],
            t=record["t"],
            kp=record["kp"]
        ))
    
    return Observation(points=points)
