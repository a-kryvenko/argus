"""Shared physical-unit OMNI missing-value contract for ingestion and training."""
import pandas as pd

# Hourly OMNI2 fill values; Kp in persisted CSVs is already divided by ten.
# https://omniweb.gsfc.nasa.gov/html/ow_data.html
OMNI_FILL_VALUES = {
    "bx": 999.9, "by": 999.9, "bz": 999.9, "t": 9999999,
    "n": 999.9, "v": 9999, "kp": 9.9, "dst": 99999,
    "ap": 999, "f10_7": 999.9,
}


def clean_omni_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean physical-unit OMNI columns without filling missing observations."""
    frame = frame.copy()
    for column, fill in OMNI_FILL_VALUES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = values.mask(values.abs().eq(fill))
    return frame

