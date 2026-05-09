import requests
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from common.config import get_config
from common.schema import ObservationPoint, Observation

DSCOVR_2H_PLASMA_BASE_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-"
DSCOVR_2H_MAG_BASE_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-"

def _fetch_latest_observations(endpoint: str) -> pd.DataFrame:
    r = requests.get(DSCOVR_2H_MAG_BASE_URL + endpoint)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df["issue_time"] = pd.to_datetime(df["time_tag"])
    df["bx"] = pd.to_numeric(df["bx_gsm"])
    df["by"] = pd.to_numeric(df["by_gsm"])
    df["bz"] = pd.to_numeric(df["bz_gsm"])
    df = df.set_index("issue_time", drop=False)

    r = requests.get(DSCOVR_2H_PLASMA_BASE_URL + endpoint)
    r.raise_for_status()
    data = r.json()

    plasma_df = pd.DataFrame(data[1:], columns=data[0])
    plasma_df["issue_time"] = pd.to_datetime(plasma_df["time_tag"])
    plasma_df["n"] = pd.to_numeric(plasma_df["density"])
    plasma_df["v"] = pd.to_numeric(plasma_df["speed"])
    plasma_df["t"] = pd.to_numeric(plasma_df["temperature"])
    plasma_df = plasma_df.set_index("issue_time", drop=False)

    df = df.merge(plasma_df, how="left", on="issue_time")

    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.resample("1h").mean()

    return df

    
def _get_raw_dataset(raw_dataset_path: Path) -> pd.DataFrame:
    now = datetime.now(timezone.utc)

    df = None

    if raw_dataset_path.is_file():
        df = pd.read_csv(raw_dataset_path, parse_dates=["issue_time"])
        time_delta = now - df.iloc[-1]["issue_time"]

        # We are able to get only up to 6 day live observations directly from NOAA json products.
        # For older data we must explore FTP server
        if time_delta.total_seconds() > 3600 * 24 * 6:
            df = None
    
    if df is None:
        df = _download_raw_dataset(raw_dataset_path)
    
    time_delta = now - df.iloc[-1]["issue_time"]

    if time_delta < 60 * 5:
        endpoint = "5-minute.json"
    elif time_delta < 3600 * 2:
        endpoint = "2-hour.json"
    elif time_delta < 3600 * 6:
        endpoint = "6-hour.json"
    elif time_delta < 3600 * 24:
        endpoint = "1-day.json"
    else:
        endpoint = "7-day.json"
    
    latest_df = _fetch_latest_observations(endpoint=endpoint)

    df = pd.concat([df, latest_df])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.resample("1h").last()

    df.to_csv(raw_dataset_path, index=False)

    return df


def _download_raw_dataset(raw_dataset_path: Path) -> pd.DataFrame:
    # go to FTP, download latest dataset for 1 month back, save to raw_dataset_path and return dataframe
    pass


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
            kp=0 # TODO: fil lwith actual values
        ))
    
    return Observation(points=points)
