"""Observed F10.7 flux from NOAA SWPC, preserving source timestamps."""
import pandas as pd
import requests

F10_7_FLUX_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"


def fetch_f107() -> pd.DataFrame:
    r = requests.get(F10_7_FLUX_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    flux_df = pd.DataFrame(data, columns=["time_tag", "flux"])
    flux_df["issue_time"] = pd.to_datetime(flux_df["time_tag"], utc=True)
    flux_df["f10_7"] = pd.to_numeric(flux_df["flux"], errors="coerce")
    return flux_df[["issue_time", "f10_7"]]
