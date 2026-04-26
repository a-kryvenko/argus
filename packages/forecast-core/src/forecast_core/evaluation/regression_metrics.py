from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


COMPONENTS = ["Bx", "By", "Bz", "N"]


def summarize_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(COMPONENTS):
        truth = y_true[..., idx].ravel()
        pred = y_pred[..., idx].ravel()
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
            "mae": float(mean_absolute_error(truth, pred)),
            "r2": float(r2_score(truth, pred)),
        }
    return metrics
