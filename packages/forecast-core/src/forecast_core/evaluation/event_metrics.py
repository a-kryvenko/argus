from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def summarize_bz_event_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    flat_true = y_true.ravel()
    flat_probs = probs.ravel()
    if np.unique(flat_true).size < 2:
        return {"auroc": float("nan"), "average_precision": float("nan")}
    return {
        "auroc": float(roc_auc_score(flat_true, flat_probs)),
        "average_precision": float(average_precision_score(flat_true, flat_probs)),
    }
