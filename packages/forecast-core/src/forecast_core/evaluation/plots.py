from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_forecast_example(y_true: np.ndarray, y_pred: np.ndarray, sample_idx: int = 0):
    names = ["Bx", "By", "Bz", "N"]
    steps = np.arange(1, y_true.shape[1] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, idx, name in zip(axes.flatten(), range(4), names, strict=False):
        ax.plot(steps, y_true[sample_idx, :, idx], label="target")
        ax.plot(steps, y_pred[sample_idx, :, idx], label="pred")
        ax.set_title(name)
    fig.tight_layout()
    return fig
