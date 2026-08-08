import pandas as pd
import numpy as np

SOLAR_ROTATION_DAYS = 27.2753

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 6:
        raise Exception("Dataset size must be greater than 6 hours to have persistant t-6h speed observation")
    
    df["bt"] = np.sqrt(df["bx"]**2 + df["by"]**2 + df["bz"]**2)
    df["southward_bz"] = np.maximum(-df["bz"], 0)
    df["bz_over_bt"] = np.minimum(df["bz"] / df["bt"], 1)
    df["dynamic_pressure"] = df["n"] * df["v"]**2

    feature_windows = {
        "v": [3, 6, 168],
        "bz": [3, 6, 168],
        "bt": [3, 6, 168],
        "n": [3, 6, 168],
        "t": [3, 6, 168],
        "dynamic_pressure": [3, 6, 168],
        "kp": [3, 6, 168],
        "ap": [3, 6, 168],
        "dst": [3, 6, 168],
        "f10_7": [3, 6, 168],
    }

    for col, windows in feature_windows.items():
        for w in windows:
            suffix = "7d" if w == 168 else f"{w}h"
            mean_col = f"{col}_mean_{suffix}"
            delta_col = f"{col}_delta_{suffix}"

            df[mean_col] = df[col].rolling(window=w, min_periods=1).mean()

            df[delta_col] = df[col] - df[mean_col]

    return df
