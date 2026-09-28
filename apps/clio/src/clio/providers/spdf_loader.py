from datetime import datetime, timedelta
import requests

import pandas as pd
import cdflib
import tempfile

SOLAR_WIND_URL_PATTERN = "https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_k0/{year}/ac_k0_swe_{ymd}_v01.cdf"
MAGNETIC_URL_PATTERN = "https://spdf.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_k0/{year}/ac_k0_mfi_{ymd}_v01.cdf"

class SPDF_Loader:
    @staticmethod
    def load_plasma(start_date: datetime, end_date: datetime):
        """Hourly plasma samples, independent of magnetic archive availability."""
        frames = []
        day = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= end_date:
            cdf = SPDF_Loader._load_cdf_from_url(SOLAR_WIND_URL_PATTERN.format(
                year=day.strftime('%Y'), ymd=day.strftime('%Y%m%d')))
            if cdf is not None:
                frames.append(SPDF_Loader._swepam_dataframe(cdf))
            day += timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=['issue_time', 'v', 'n', 't'])
        frame = pd.concat(frames, ignore_index=True).sort_values('issue_time')
        frame = frame.drop_duplicates('issue_time', keep='last').set_index('issue_time')
        # Reject fill values before selecting an hourly sample; do not allow a
        # sentinel to mask a later valid observation in the same hour.
        frame[['v', 'n', 't']] = frame[['v', 'n', 't']].where(frame[['v', 'n', 't']] >= 0)
        return frame.resample('1h').first().reset_index()

    def load(start_date: datetime, end_date: datetime):
        mag_frames = []
        swe_frames = []
    
        d = start_date
    
        while d <= end_date:
            year = d.strftime("%Y")
            ymd = d.strftime("%Y%m%d")

            swe_cdf = SPDF_Loader._load_cdf_from_url(SOLAR_WIND_URL_PATTERN.format(year=year, ymd=ymd))
            mag_cdf = SPDF_Loader._load_cdf_from_url(MAGNETIC_URL_PATTERN.format(year=year, ymd=ymd))
    
            if mag_cdf and swe_cdf:
                swe_frames.append(
                    SPDF_Loader._swepam_dataframe(swe_cdf)
                )
                mag_frames.append(
                    SPDF_Loader._mag_dataframe(mag_cdf)
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
        df = df.dropna(subset=["issue_time"])
        df = df.set_index("issue_time", drop=False)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df.resample("1h").first()
    
        return df

    def _load_cdf_from_url(url: str):
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            print("missing:", url)
            return None
        
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".cdf", delete=True)
        tmp.write(r.content)
        tmp.flush()

        return cdflib.CDF(tmp.name)
    
    def _mag_dataframe(cdf):
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


    def _swepam_dataframe(cdf):
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
