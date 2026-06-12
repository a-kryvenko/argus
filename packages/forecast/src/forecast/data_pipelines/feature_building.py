import pandas as pd
import numpy as np


SOLAR_ROTATION_DAYS = 27.2753

BASE_FEATURE_COLUMNS = [
    "v_obs",
    "n",
    "bz",
    "bt",
    "kp",

    "v_persist_1h",
    "v_persist_6h",
    "v_persist_24h",
    "v_persist_27d",

    "delta_v_1h_6h",
    "delta_v_1h_24h",
    "delta_v_24h_27d",

    "abs_bz",
    "southward_bz",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 27 * 24:
        raise Exception("Dataset size must be greater than 27 days to have persistant t-27d speed observation")

    df["v_obs"] = df["v"]
    
    df["v_persist_1h"] = df["v_obs"].shift(1)
    df["v_persist_6h"] = df["v_obs"].shift(6)
    df["v_persist_24h"] = df["v_obs"].shift(24)
    df["v_persist_27d"] = df["v_obs"].shift(27 * 24)

    df["delta_v_1h_6h"] = df["v_persist_1h"] - df["v_persist_6h"]
    df["delta_v_1h_24h"] = df["v_persist_1h"] - df["v_persist_24h"]
    df["delta_v_24h_27d"] = df["v_persist_24h"] - df["v_persist_27d"]

    df["bt"] = np.sqrt(
        df["bx"] ** 2 +
        df["by"] ** 2 +
        df["bz"] ** 2
    )

    df["abs_bz"] = df["bz"].abs()
    df["southward_bz"] = np.maximum(0.0, -df["bz"])

    unix_hours = df.index.view("int64") / 1e9 / 3600.0
    period_hours = SOLAR_ROTATION_DAYS * 24.0

    df["issue_sin_27d"] = np.sin(2.0 * np.pi * unix_hours / period_hours)
    df["issue_cos_27d"] = np.cos(2.0 * np.pi * unix_hours / period_hours)

    df = df.reset_index()

    keep = ["issue_time"] + BASE_FEATURE_COLUMNS

    df = df[keep]

    df = df.dropna(
        subset=[
            "v_obs",
            "v_persist_1h",
            "v_persist_6h",
            "v_persist_24h",
            "v_persist_27d",
        ]
    )

    return df
