from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_engineering import TARGET_COLUMNS


def build_targets(df: pd.DataFrame, start_idx: int, output_steps: int) -> np.ndarray:
    rows = df.iloc[start_idx + 1:start_idx + 1 + output_steps]
    return rows[TARGET_COLUMNS].to_numpy(dtype=np.float32)


def build_bz_event_targets(df: pd.DataFrame, start_idx: int, output_steps: int, threshold: float = -5.0) -> np.ndarray:
    rows = df.iloc[start_idx + 1:start_idx + 1 + output_steps]
    return (rows["BZ_GSM"].to_numpy(dtype=np.float32) < threshold).astype(np.float32)
