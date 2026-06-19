import pandas as pd
import numpy as np

SOLAR_ROTATION_DAYS = 27.2753

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 27 * 24:
        raise Exception("Dataset size must be greater than 27 days to have persistant t-27d speed observation")
    
    df["bt"] = np.sqrt(df["bx"]**2 + df["by"]**2 + df["bz"]**2)
    df["southward_bz"] = np.maximum(-df["bz"], 0)
    df["bz_over_bt"] = np.minimum(df["bz"] / df["bt"], 1)
    df["dynamic_pressure"] = df["n"] * df["v"]**2

    cols = ["v", "bz", "bt", "n", "t", "dynamic_pressure", "kp"]
    windows = [3, 6]

    for col in cols:
        for w in windows:
            mean_col = f"{col}_mean_{w}h"
            delta_col = f"{col}_delta_{w}h"

            df[mean_col] = df[col].rolling(window=w, min_periods=1).mean()

            df[delta_col] = df[col] - df[mean_col]

    return df
