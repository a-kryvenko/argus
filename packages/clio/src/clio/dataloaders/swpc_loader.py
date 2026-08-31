from datetime import datetime

import pandas as pd
import requests

L1_SENSORS_URL = "https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind.json"
KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
SOLAR_CYCLE_INFO_URL = "https://services.swpc.noaa.gov/products/solar-cycle-25-f10-7-predicted-range.json"
F10_7_FLUX_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
DST_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"


class SWPC_Loader:
    METRICS = ("bx", "by", "bz", "v", "n", "t", "kp", "dst", "ap", "f10_7")

    def load(start_date: datetime) -> pd.DataFrame:
        """Return the legacy hourly wide frame without persisting it."""

        frames = SWPC_Loader._fetch_source_frames()
        df = SWPC_Loader._hourly(frames[0])
        for frame in frames[1:]:
            df = df.merge(
                SWPC_Loader._hourly(frame),
                how="left",
                on="issue_time",
            )

        df = df[df["issue_time"] > start_date]

        return df

    @staticmethod
    def load_measurements(start_date: datetime | None = None) -> pd.DataFrame:
        """Fetch source records in the narrow raw-measurement format."""

        frames = []
        for frame in SWPC_Loader._fetch_source_frames():
            value_columns = [column for column in SWPC_Loader.METRICS if column in frame]
            narrow = frame.melt(
                id_vars="issue_time",
                value_vars=value_columns,
                var_name="metric",
                value_name="value",
            ).rename(columns={"issue_time": "observed_at"})
            frames.append(narrow)

        measurements = pd.concat(frames, ignore_index=True)
        measurements["observed_at"] = pd.to_datetime(
            measurements["observed_at"],
            utc=True,
        )
        measurements["value"] = pd.to_numeric(
            measurements["value"],
            errors="coerce",
        )
        measurements = measurements.dropna(subset=["observed_at", "value"])
        if start_date is not None:
            cutoff = pd.Timestamp(start_date)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tz is None else cutoff.tz_convert("UTC")
            measurements = measurements[measurements["observed_at"] > cutoff]

        return measurements.sort_values(["observed_at", "metric"]).reset_index(drop=True)

    @staticmethod
    def _fetch_source_frames() -> tuple[pd.DataFrame, ...]:
        return (
            SWPC_Loader._fetch_live_sensors(),
            SWPC_Loader._fetch_live_kp(),
            SWPC_Loader._fetch_f10_7_flux(),
            SWPC_Loader._fetch_dst(),
        )

    @staticmethod
    def _hourly(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.set_index("issue_time").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        return (
            frame
            .resample("1h")
            .first()
            .interpolate(method="time")
            .reset_index()
        )

    @staticmethod
    def _fetch_live_sensors() -> pd.DataFrame:
        r = requests.get(L1_SENSORS_URL, timeout=60)
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame(data[1:], columns=data[0])
        df["issue_time"] = pd.to_datetime(df["time_tag"], utc=True)

        df["bx"] = pd.to_numeric(df["bx"], errors="coerce")
        df["by"] = pd.to_numeric(df["by"], errors="coerce")
        df["bz"] = pd.to_numeric(df["bz"], errors="coerce")

        # df["vx"] = pd.to_numeric(df["vx"])
        # df["vy"] = pd.to_numeric(df["vy"])
        # df["vz"] = pd.to_numeric(df["vz"])

        df["n"] = pd.to_numeric(df["density"], errors="coerce")
        df["v"] = pd.to_numeric(df["speed"], errors="coerce")
        df["t"] = pd.to_numeric(df["temperature"], errors="coerce")

        return df[["issue_time", "bx", "by", "bz", "n", "v", "t"]]

    @staticmethod
    def _fetch_live_kp() -> pd.DataFrame:
        r = requests.get(KP_INDEX_URL, timeout=60)
        r.raise_for_status()
        data = r.json()
        kp_df = pd.DataFrame(data, columns=["time_tag", "Kp", "a_running"])
        kp_df["issue_time"] = pd.to_datetime(kp_df["time_tag"], utc=True)
        kp_df["kp"] = pd.to_numeric(kp_df["Kp"], errors="coerce")
        kp_df["ap"] = pd.to_numeric(kp_df["a_running"], errors="coerce")
        return kp_df[["issue_time", "kp", "ap"]]

    @staticmethod
    def _fetch_f10_7_flux() -> pd.DataFrame:
        r = requests.get(F10_7_FLUX_URL, timeout=60)
        r.raise_for_status()
        data = r.json()
        flux_df = pd.DataFrame(data, columns=["time_tag", "flux"])
        flux_df["issue_time"] = pd.to_datetime(flux_df["time_tag"], utc=True)
        flux_df["f10_7"] = pd.to_numeric(flux_df["flux"], errors="coerce")
        return flux_df[["issue_time", "f10_7"]]

    @staticmethod
    def _fetch_dst() -> pd.DataFrame:
        r = requests.get(DST_URL, timeout=60)
        r.raise_for_status()
        data = r.json()
        dst_df = pd.DataFrame(data, columns=["time_tag", "dst"])
        dst_df["issue_time"] = pd.to_datetime(dst_df["time_tag"], utc=True)
        dst_df["dst"] = pd.to_numeric(dst_df["dst"], errors="coerce")
        return dst_df[["issue_time", "dst"]]
