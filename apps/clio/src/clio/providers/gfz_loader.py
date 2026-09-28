"""Observed daily solar flux from GFZ."""
from io import StringIO

import numpy as np
import pandas as pd
import requests

GFZ_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"


def parse_gfz_f107(text: str) -> pd.DataFrame:
    """GFZ daily format: date, indices, F10.7 observed/adjusted, definitive flag.

    Use observed flux (third column from the end), not adjusted flux or Ap.
    Format: https://kp.gfz.de/en/data
    """
    raw = pd.read_csv(StringIO(text), sep=r"\s+", comment="#", header=None)
    if raw.shape[1] != 28:
        raise ValueError("Unexpected GFZ daily-index format (expected 28 columns)")
    dates = pd.to_datetime(dict(year=raw[0], month=raw[1], day=raw[2]), utc=True)
    values = pd.to_numeric(raw[25], errors="coerce")
    valid = np.isfinite(values) & values.gt(0)
    return pd.DataFrame({"metric": "f10_7", "value": values[valid],
                         "observed_at": dates[valid] + pd.Timedelta(hours=12)})



def load_gfz_f107(start, end) -> pd.DataFrame:
    response = requests.get(GFZ_URL, timeout=120)
    response.raise_for_status()
    frame = parse_gfz_f107(response.text)
    return frame.loc[frame.observed_at.between(start, end)]
