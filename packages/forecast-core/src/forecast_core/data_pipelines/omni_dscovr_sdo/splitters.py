from __future__ import annotations

import pandas as pd


def blocked_train_valid_split(df: pd.DataFrame, valid_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = int(len(df) * (1.0 - valid_fraction))
    return df.iloc[:pivot].copy(), df.iloc[pivot:].copy()
