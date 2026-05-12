import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

FEATURE_COLUMNS = [
    "lead_hours", "lead_norm",
    "v_obs", "n_obs", "bz_obs", "bt_obs", "kp",
    "v_persist_1h", "v_persist_6h", "v_persist_24h", "v_persist_27d",
    "delta_v_1h_6h", "delta_v_1h_24h", "delta_v_24h_27d",
    "abs_bz", "southward_bz",
]

THRESHOLDS = [450, 500, 600, 700]

def main():
    parser = argparse.ArgumentParser(
        description="Train wind speed threshold event probability models"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path
    )
    parser.add_argument(
        "--model-out",
        default=Path("models/speed_events.joblib"),
        type=Path
    )

    df = pd.read_csv("../../data/historical/omni_processed_2020_2023.csv")

    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

    df = df.dropna(subset=["target_v", *FEATURE_COLUMNS])
    df = df.sort_values(["issue_time", "lead_hours"])

    for threshold in THRESHOLDS:
        df[f"target_v_ge_{threshold}"] = (df["target_v"] >= threshold).astype(int)


if __name__ == "__main__":
    main()