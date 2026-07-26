from datetime import datetime

import pandas as pd

import requests

L1_SENSORS_URL = "https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind.json"
KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SOLAR_CYCLE_INFO_URL = "https://services.swpc.noaa.gov/products/solar-cycle-25-f10-7-predicted-range.json"
F10_7_FLUX_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
DST_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"

class SWPC_Loader:
    def load(start_date: datetime) -> pd.DataFrame:
        df = SWPC_Loader._fetch_live_sensors()
        df = df.merge(SWPC_Loader._fetch_live_kp(), how="left", on="issue_time")
        df = df.merge(SWPC_Loader._fetch_f10_7_flux(), how="left", on="issue_time")
        df = df.merge(SWPC_Loader._fetch_dst(), how="left", on="issue_time")
    
        df = df.set_index("issue_time", drop=False)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df.resample("1h").first()

        df = df[df["issue_time"] > start_date]
    
        return df

    def _fetch_live_sensors() -> pd.DataFrame:
        r = requests.get(L1_SENSORS_URL)
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame(data[1:], columns=data[0])
        df["issue_time"] = pd.to_datetime(df["time_tag"])

        df["bx"] = pd.to_numeric(df["bx"])
        df["by"] = pd.to_numeric(df["by"])
        df["bz"] = pd.to_numeric(df["bz"])

        # df["vx"] = pd.to_numeric(df["vx"])
        # df["vy"] = pd.to_numeric(df["vy"])
        # df["vz"] = pd.to_numeric(df["vz"])

        df["n"] = pd.to_numeric(df["density"])
        df["v"] = pd.to_numeric(df["speed"])
        df["t"] = pd.to_numeric(df["temperature"])

        return df

    def _fetch_live_kp() -> pd.DataFrame:
        r = requests.get(KP_INDEX_URL)
        r.raise_for_status()
        data = r.json()
        kp_df = pd.DataFrame(data, columns=["time_tag", "Kp", "a_running"])
        kp_df["issue_time"] = pd.to_datetime(kp_df["time_tag"])
        kp_df["issue_time"] = kp_df["issue_time"].dt.tz_localize("UTC")
        kp_df["kp"] = pd.to_numeric(kp_df["Kp"])
        kp_df["ap"] = pd.to_numeric(kp_df["a_running"])
        kp_df = kp_df[["issue_time", "kp", "ap"]]
        kp_df = kp_df.set_index("issue_time")

        kp_df = (
            kp_df
            .resample("1h")
            .interpolate(method="time")
            .reset_index()
        )

        return kp_df

    def _fetch_f10_7_flux() -> pd.DataFrame:
        r = requests.get(F10_7_FLUX_URL)
        r.raise_for_status()
        data = r.json()
        flux_df = pd.DataFrame(data, columns=["time_tag", "flux"])
        flux_df["issue_time"] = pd.to_datetime(flux_df["time_tag"])
        flux_df["issue_time"] = flux_df["issue_time"].dt.tz_localize("UTC")
        flux_df["f10_7"] = pd.to_numeric(flux_df["flux"])
        flux_df = flux_df[["issue_time", "f10_7"]]
        flux_df = flux_df.set_index("issue_time")

        flux_df = (
            flux_df
            .resample("1h")
            .interpolate(method="time")
            .reset_index()
        )

        return flux_df

    def _fetch_dst() -> pd.DataFrame:
        r = requests.get(DST_URL)
        r.raise_for_status()
        data = r.json()
        dst_df = pd.DataFrame(data, columns=["time_tag", "dst"])
        dst_df["issue_time"] = pd.to_datetime(dst_df["time_tag"])
        dst_df["issue_time"] = dst_df["issue_time"].dt.tz_localize("UTC")
        dst_df["dst"] = pd.to_numeric(dst_df["dst"])
        dst_df = dst_df[["issue_time", "dst"]]
        dst_df = dst_df.set_index("issue_time")

        dst_df = (
            dst_df
            .resample("1h")
            .interpolate(method="time")
            .reset_index()
        )

        return dst_df

