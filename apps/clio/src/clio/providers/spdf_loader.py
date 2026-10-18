from datetime import datetime, timedelta
import requests

import pandas as pd
import numpy as np
import cdflib
import tempfile

SOLAR_WIND_URL_PATTERN = "https://spdf.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_k0/{year}/ac_k0_swe_{ymd}_v01.cdf"

class SPDF_Loader:
    @staticmethod
    def _values(cdf, name):
        """Mask CDF fill values and validity bounds before hourly resampling."""
        values = np.asarray(cdf.varget(name), dtype=float)
        attrs = cdf.varattsget(name)
        valid = np.isfinite(values)
        if 'FILLVAL' in attrs:
            valid &= values != attrs['FILLVAL']
        if 'VALIDMIN' in attrs:
            valid &= values >= attrs['VALIDMIN']
        if 'VALIDMAX' in attrs:
            valid &= values <= attrs['VALIDMAX']
        return np.where(valid, values, np.nan)

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
    
    def _swepam_dataframe(cdf):
        times = cdflib.cdfepoch.to_datetime(
            cdf.varget("Epoch")
        )
        df = pd.DataFrame({
            "issue_time": times,
            "v": SPDF_Loader._values(cdf, "Vp"),
            "n": SPDF_Loader._values(cdf, "Np"),
            "t": SPDF_Loader._values(cdf, "Tpr")
        })
        df["issue_time"] = df["issue_time"].dt.tz_localize("UTC")
        return df
