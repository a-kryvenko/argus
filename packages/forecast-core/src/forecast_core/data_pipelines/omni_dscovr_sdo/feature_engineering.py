from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["V", "BX_GSE", "BY_GSM", "BZ_GSM", "N"]
TARGET_COLUMNS = ["BX_GSE", "BY_GSM", "BZ_GSM", "N"]


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bz_is_southward"] = (out["BZ_GSM"] < 0).astype(np.float32)
    out["bz_lt_minus_5"] = (out["BZ_GSM"] < -5).astype(np.float32)
    out["dynamic_pressure_proxy"] = out["N"].clip(lower=0) * out["V"].clip(lower=0).pow(2)
    out["bz_rolling_3h_mean"] = out["BZ_GSM"].rolling(3, min_periods=1).mean()
    out["bz_rolling_6h_mean"] = out["BZ_GSM"].rolling(6, min_periods=1).mean()
    out["speed_rolling_6h_mean"] = out["V"].rolling(6, min_periods=1).mean()
    out["density_rolling_6h_mean"] = out["N"].rolling(6, min_periods=1).mean()
    return out


def build_context_vector(df: pd.DataFrame, idx: int) -> np.ndarray:
    row = df.iloc[idx]
    return np.array([
        row["bz_is_southward"],
        row["bz_lt_minus_5"],
        row["dynamic_pressure_proxy"],
        row["bz_rolling_3h_mean"],
        row["bz_rolling_6h_mean"],
        row["speed_rolling_6h_mean"],
        row["density_rolling_6h_mean"],
    ], dtype=np.float32)
